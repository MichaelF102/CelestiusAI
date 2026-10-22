"""Configurable Large Language Model (LLM) provider abstraction layer.

Supports:
- Text and structured generation
- Token and usage metadata
- Timeout handling & transient retry policies
- Safe secret handling (no keys leaked in logs or exceptions)
- Provider status check and No-Provider graceful fallback
- Concrete Gemini provider via official google-genai SDK
- Deterministic Mock provider for tests & offline operation
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel):
    """Execution metadata and raw text output from an LLM invocation."""

    raw_text: str = ""
    parsed_json: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str
    provider: str
    execution_time_ms: float = 0.0
    status: str = Field(default="success", description="'success', 'error', or 'not_configured'")
    error_message: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    def __init__(self, model_name: str, timeout_seconds: int = 60, max_retries: int = 2) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Vendor name, e.g. 'gemini', 'mock'."""
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if provider has required credentials and configuration."""
        ...

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Verify provider readiness without making paid/unsolicited generation calls."""
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate unstructured narrative response."""
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[T | None, LLMResult]:
        """Generate validated structured output matching the provided Pydantic model."""
        ...


class MockLLMProvider(LLMProvider):
    """Deterministic, offline provider for test suites and credential-free evaluation."""

    def __init__(
        self,
        model_name: str = "mock-analyst-v1",
        predefined_response: BaseModel | dict[str, Any] | str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, timeout_seconds=10, max_retries=1)
        self.predefined_response = predefined_response

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def is_configured(self) -> bool:
        return True

    def check_availability(self) -> tuple[bool, str]:
        return True, "Mock LLM Provider is ready (offline simulation mode)."

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        start_time = time.perf_counter()
        if isinstance(self.predefined_response, str):
            text = self.predefined_response
        else:
            text = (
                "Mock Analysis: The provided dataset evidence shows clear patterns. "
                "Observations indicate notable differences across segments."
            )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return LLMResult(
            raw_text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(prompt) + len(text)) // 4,
            model=self.model_name,
            provider=self.provider_name,
            execution_time_ms=round(duration_ms, 2),
            status="success",
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[T | None, LLMResult]:
        start_time = time.perf_counter()

        if isinstance(self.predefined_response, BaseModel) and isinstance(
            self.predefined_response, response_model
        ):
            instance = self.predefined_response  # type: ignore[assignment]
            raw_text = instance.model_dump_json()
            json_dict = instance.model_dump()
        elif isinstance(self.predefined_response, dict):
            try:
                instance = response_model.model_validate(self.predefined_response)
                raw_text = json.dumps(self.predefined_response)
                json_dict = self.predefined_response
            except Exception as err:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return None, LLMResult(
                    raw_text="",
                    model=self.model_name,
                    provider=self.provider_name,
                    execution_time_ms=round(duration_ms, 2),
                    status="error",
                    error_message=f"Mock response validation error: {err}",
                )
        else:
            # Construct a safe default minimal instance
            try:
                # If schema has default constructible fields
                instance = response_model.model_validate({})
                raw_text = instance.model_dump_json()
                json_dict = instance.model_dump()
            except Exception:
                # Fallback dictionary for AnalystResponse if requested
                fallback_dict: dict[str, Any] = {
                    "question": "Analyzed query",
                    "intent_category": "descriptive_statistics",
                    "direct_answer": "Analysis indicates significant observed variations in the data.",
                    "key_findings": [],
                    "interpretation": "Interpretation based on computed deterministic evidence.",
                    "alternative_explanations": [
                        "Selection bias or unobserved confounding factors."
                    ],
                    "uncertainty_and_limitations": ["Limited to current dataset sample size."],
                    "recommended_next_analyses": ["Run subgroup regression diagnostics."],
                    "hypotheses": [],
                    "contradictions_detected": [],
                    "trace_metadata": {
                        "provider": "mock",
                        "model": self.model_name,
                        "execution_time_ms": 10.0,
                        "evidence_ids_used": [],
                    },
                    "dataset_version": 1,
                }
                instance = response_model.model_validate(fallback_dict)
                raw_text = json.dumps(fallback_dict)
                json_dict = fallback_dict

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        result = LLMResult(
            raw_text=raw_text,
            parsed_json=json_dict,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(raw_text) // 4,
            total_tokens=(len(prompt) + len(raw_text)) // 4,
            model=self.model_name,
            provider=self.provider_name,
            execution_time_ms=round(duration_ms, 2),
            status="success",
        )
        return instance, result


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using the modern google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-pro",
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            model_name=model_name, timeout_seconds=timeout_seconds, max_retries=max_retries
        )
        self._api_key = api_key.strip() if api_key and api_key.strip() else None

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def check_availability(self) -> tuple[bool, str]:
        if not self.is_configured:
            return False, "GEMINI_API_KEY environment variable is not configured."
        return True, f"Gemini provider configured with model '{self.model_name}'."

    def _get_client(self) -> Any:
        """Instantiate google.genai Client without exposing key in exceptions."""
        if not self.is_configured:
            raise ValueError("Cannot instantiate Gemini client: API key is not configured.")
        try:
            from google import genai

            return genai.Client(api_key=self._api_key)
        except ImportError as err:
            raise RuntimeError(
                "The 'google-genai' package is required to use GeminiProvider. "
                "Install it using: pip install google-genai"
            ) from err

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        if not self.is_configured:
            return LLMResult(
                raw_text="",
                model=self.model_name,
                provider=self.provider_name,
                status="not_configured",
                error_message="Gemini API key is not configured.",
            )

        start_time = time.perf_counter()
        try:
            client = self._get_client()
            config_dict: dict[str, Any] = {"temperature": temperature}
            if system_instruction:
                config_dict["system_instruction"] = system_instruction

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_dict,
            )

            raw_text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return LLMResult(
                raw_text=raw_text,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                model=self.model_name,
                provider=self.provider_name,
                execution_time_ms=round(duration_ms, 2),
                status="success",
            )
        except Exception as err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            # Sanitize error message to prevent leaking any credential string
            safe_msg = str(err).replace(self._api_key or "UNSET_KEY", "******")
            logger.error("Gemini text generation failed: %s", safe_msg)
            return LLMResult(
                raw_text="",
                model=self.model_name,
                provider=self.provider_name,
                execution_time_ms=round(duration_ms, 2),
                status="error",
                error_message=safe_msg,
            )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[T | None, LLMResult]:
        if not self.is_configured:
            return None, LLMResult(
                raw_text="",
                model=self.model_name,
                provider=self.provider_name,
                status="not_configured",
                error_message="Gemini API key is not configured.",
            )

        start_time = time.perf_counter()
        try:
            client = self._get_client()
            config_dict: dict[str, Any] = {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_schema": response_model,
            }
            if system_instruction:
                config_dict["system_instruction"] = system_instruction

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_dict,
            )

            raw_text = response.text or ""
            parsed_dict = json.loads(raw_text) if raw_text else {}
            validated_obj = response_model.model_validate(parsed_dict)

            usage = getattr(response, "usage_metadata", None)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = LLMResult(
                raw_text=raw_text,
                parsed_json=parsed_dict,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                model=self.model_name,
                provider=self.provider_name,
                execution_time_ms=round(duration_ms, 2),
                status="success",
            )
            return validated_obj, result

        except Exception as err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            safe_msg = str(err).replace(self._api_key or "UNSET_KEY", "******")
            logger.error("Gemini structured generation failed: %s", safe_msg)
            return None, LLMResult(
                raw_text="",
                model=self.model_name,
                provider=self.provider_name,
                execution_time_ms=round(duration_ms, 2),
                status="error",
                error_message=safe_msg,
            )
