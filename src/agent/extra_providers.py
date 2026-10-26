"""Additional LLM backends: Anthropic (Claude) and Ollama (local models).

Both share one implementation of text/structured generation on top of a single
``_complete`` call, so adding another backend only needs that method.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import abstractmethod
from typing import Any

from src.agent.llm_provider import LLMProvider, LLMResult, T

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class _CompletionProvider(LLMProvider):
    """Template: subclasses implement `_complete` returning (text, usage dict)."""

    secret: str | None = None

    @abstractmethod
    def _complete(
        self, prompt: str, system_instruction: str | None, temperature: float, json_mode: bool
    ) -> tuple[str, dict[str, int | None]]: ...

    def _scrub(self, message: str) -> str:
        return message.replace(self.secret, "******") if self.secret else message

    def _error(self, start: float, message: str, status: str = "error") -> LLMResult:
        return LLMResult(
            raw_text="",
            model=self.model_name,
            provider=self.provider_name,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            status=status,
            error_message=self._scrub(message),
        )

    def generate_text(
        self, prompt: str, system_instruction: str | None = None, temperature: float = 0.2
    ) -> LLMResult:
        start = time.perf_counter()
        if not self.is_configured:
            return self._error(start, f"{self.provider_name} is not configured.", "not_configured")
        try:
            text, usage = self._complete(prompt, system_instruction, temperature, json_mode=False)
        except Exception as err:
            logger.error("%s text generation failed: %s", self.provider_name, self._scrub(str(err)))
            return self._error(start, str(err))
        return LLMResult(
            raw_text=text,
            prompt_tokens=usage.get("prompt"),
            completion_tokens=usage.get("completion"),
            total_tokens=(usage.get("prompt") or 0) + (usage.get("completion") or 0) or None,
            model=self.model_name,
            provider=self.provider_name,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[T | None, LLMResult]:
        start = time.perf_counter()
        if not self.is_configured:
            return None, self._error(
                start, f"{self.provider_name} is not configured.", "not_configured"
            )
        schema = json.dumps(response_model.model_json_schema())
        full_prompt = (
            f"{prompt}\n\nRespond with ONE JSON object only (no prose, no code fences) that "
            f"validates against this JSON Schema:\n{schema}"
        )
        try:
            text, usage = self._complete(
                full_prompt, system_instruction, temperature, json_mode=True
            )
            match = _JSON_OBJECT.search(text)
            parsed = json.loads(match.group(0) if match else text)
            validated = response_model.model_validate(parsed)
        except Exception as err:
            logger.error(
                "%s structured generation failed: %s", self.provider_name, self._scrub(str(err))
            )
            return None, self._error(start, str(err))
        return validated, LLMResult(
            raw_text=text,
            parsed_json=parsed,
            prompt_tokens=usage.get("prompt"),
            completion_tokens=usage.get("completion"),
            model=self.model_name,
            provider=self.provider_name,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )


class AnthropicProvider(_CompletionProvider):
    """Anthropic Messages API (requires `pip install anthropic`)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-sonnet-5-5",
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries
        )
        self.secret = api_key.strip() if api_key and api_key.strip() else None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def is_configured(self) -> bool:
        return bool(self.secret)

    def check_availability(self) -> tuple[bool, str]:
        if not self.is_configured:
            return False, "ANTHROPIC_API_KEY is not configured."
        return True, f"Anthropic provider configured with model '{self.model_name}'."

    def _complete(self, prompt, system_instruction, temperature, json_mode):  # type: ignore[no-untyped-def]
        try:
            import anthropic
        except ImportError as err:
            raise RuntimeError("Install the SDK first: pip install anthropic") from err
        client = anthropic.Anthropic(
            api_key=self.secret, timeout=self.timeout_seconds, max_retries=self.max_retries
        )
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        response = client.messages.create(**kwargs)
        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = {"prompt": response.usage.input_tokens, "completion": response.usage.output_tokens}
        return text, usage


class OllamaProvider(_CompletionProvider):
    """Local models served by Ollama (default http://localhost:11434); no key required."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        max_retries: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries
        )
        self.host = host.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def is_configured(self) -> bool:
        return bool(self.model_name and self.host)

    def check_availability(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:  # noqa: S310
                names = [m["name"] for m in json.load(resp).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError) as err:
            return False, f"Cannot reach Ollama at {self.host}: {err}"
        if not any(n.split(":")[0] == self.model_name.split(":")[0] for n in names):
            return (
                False,
                f"Ollama is running but model '{self.model_name}' is not pulled (ollama pull {self.model_name}).",
            )
        return True, f"Ollama ready with model '{self.model_name}'."

    def _complete(self, prompt, system_instruction, temperature, json_mode):  # type: ignore[no-untyped-def]
        messages = (
            [{"role": "system", "content": system_instruction}] if system_instruction else []
        ) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"
        request = urllib.request.Request(  # noqa: S310
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:  # noqa: S310
            data = json.load(resp)
        usage = {"prompt": data.get("prompt_eval_count"), "completion": data.get("eval_count")}
        return data.get("message", {}).get("content", ""), usage
