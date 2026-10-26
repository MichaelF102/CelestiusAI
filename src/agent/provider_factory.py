"""Provider factory for instantiating validated LLM providers.

Ensures:
- Safe retrieval of API credentials from AppSettings / environment.
- No credential leaking in logs or exceptions.
- Graceful fallback to Mock or No-Provider mode.
"""

import logging
from typing import Literal

from src.agent.extra_providers import AnthropicProvider, OllamaProvider
from src.agent.llm_provider import GeminiProvider, LLMProvider, MockLLMProvider
from src.config.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)

ProviderType = Literal["gemini", "anthropic", "ollama", "mock", "auto"]
PROVIDER_CHOICES = ("gemini", "anthropic", "ollama", "mock")


class ProviderFactory:
    """Factory creating LLMProvider instances based on configuration."""

    @staticmethod
    def get_provider(
        provider_type: ProviderType | str | None = None,
        settings: AppSettings | None = None,
        mock_response: object = None,
    ) -> LLMProvider:
        """Instantiate appropriate LLMProvider.

        If provider_type is 'mock' or mock_response is provided, returns MockLLMProvider.
        If provider_type is 'gemini' or 'auto':
            Returns GeminiProvider configured with GEMINI_API_KEY if available.
        """
        app_settings = settings or get_settings()
        chosen = (provider_type or "auto").strip().lower()

        if chosen == "mock" or mock_response is not None:
            return MockLLMProvider(
                model_name="mock-analyst-v1",
                predefined_response=mock_response,  # type: ignore[arg-type]
            )

        if chosen == "anthropic":
            return AnthropicProvider(
                api_key=app_settings.anthropic_api_key, model_name=app_settings.anthropic_model
            )
        if chosen == "ollama":
            return OllamaProvider(
                model_name=app_settings.ollama_model, host=app_settings.ollama_host
            )

        # Gemini provider
        api_key = app_settings.gemini_api_key
        model_name = app_settings.gemini_model or "gemini-1.5-pro"

        logger.info(
            "Initializing LLM provider 'gemini' (model: %s, configured: %s)",
            model_name,
            bool(api_key),
        )

        return GeminiProvider(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=getattr(app_settings, "llm_timeout_seconds", 60),
            max_retries=getattr(app_settings, "llm_max_retries", 2),
        )

    @staticmethod
    def from_override(
        override: dict[str, str] | None, settings: AppSettings | None = None
    ) -> LLMProvider:
        """Build a provider from a session-only override {provider, model, api_key, host}."""
        app_settings = settings or get_settings()
        if not override:
            return ProviderFactory.get_provider(settings=app_settings)
        kind = override.get("provider", "auto")
        model = override.get("model") or None
        key = override.get("api_key") or None
        updates: dict[str, object] = {}
        if kind == "gemini":
            updates = {"gemini_api_key": key or app_settings.gemini_api_key}
            if model:
                updates["gemini_model"] = model
        elif kind == "anthropic":
            updates = {"anthropic_api_key": key or app_settings.anthropic_api_key}
            if model:
                updates["anthropic_model"] = model
        elif kind == "ollama":
            if model:
                updates["ollama_model"] = model
            if override.get("host"):
                updates["ollama_host"] = override["host"]
        return ProviderFactory.get_provider(kind, app_settings.model_copy(update=updates))
