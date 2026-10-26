"""Tests for typed configuration, environment overrides, validation, and secret masking."""

import pytest
from pydantic import ValidationError
from src.config.settings import AppSettings, get_settings, load_settings_from_env


def test_default_settings():
    """Verify default values when no environment variables are set."""
    settings = load_settings_from_env()

    assert settings.app_name == "CelestiusAI"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.max_upload_size_mb == 100
    assert settings.agent_max_iterations == 10
    assert settings.agent_execution_timeout_seconds == 120
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-1.5-pro"
    assert settings.is_llm_configured is False
    assert settings.masked_gemini_api_key == "Not configured"


def test_custom_environment_settings(monkeypatch: pytest.MonkeyPatch):
    """Verify custom environment values are properly parsed and typed."""
    monkeypatch.setenv("APP_NAME", "Custom Analytics Platform")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "250")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "25")
    monkeypatch.setenv("AGENT_EXECUTION_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySecretApiKey123456789")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")

    settings = load_settings_from_env()

    assert settings.app_name == "Custom Analytics Platform"
    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.max_upload_size_mb == 250
    assert settings.agent_max_iterations == 25
    assert settings.agent_execution_timeout_seconds == 300
    assert settings.gemini_api_key == "AIzaSySecretApiKey123456789"
    assert settings.is_llm_configured is True
    assert settings.masked_gemini_api_key == "AIza...6789"


def test_invalid_log_level():
    """Verify invalid log level raises ValidationError."""
    with pytest.raises(ValidationError):
        AppSettings(log_level="VERBOSE")  # type: ignore[arg-type]


def test_invalid_upload_size_bounds():
    """Verify upload size must be positive and within bounds."""
    with pytest.raises(ValidationError):
        AppSettings(max_upload_size_mb=0)

    with pytest.raises(ValidationError):
        AppSettings(max_upload_size_mb=5000)


def test_invalid_agent_iteration_bounds():
    """Verify iteration limits enforce sensible bounds."""
    with pytest.raises(ValidationError):
        AppSettings(agent_max_iterations=0)

    with pytest.raises(ValidationError):
        AppSettings(agent_max_iterations=500)


def test_secret_safe_representations():
    """Verify secret keys are never exposed in string representations or safe dicts."""
    secret_key = "AIzaSySuperSecretKey9999"
    settings = AppSettings(gemini_api_key=secret_key)

    repr_str = repr(settings)
    assert secret_key not in repr_str
    assert "is_llm_configured=True" in repr_str

    safe = settings.safe_dict()
    assert "gemini_api_key" not in safe
    assert safe["is_llm_configured"] is True
    assert safe["masked_gemini_api_key"] == "AIza...9999"


def test_short_secret_masking():
    """Verify short secret keys are fully masked."""
    settings = AppSettings(gemini_api_key="12345")
    assert settings.masked_gemini_api_key == "********"


def test_cached_get_settings():
    """Verify get_settings returns consistent cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
