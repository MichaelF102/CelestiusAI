"""Pytest configuration, fixtures, and environment isolation."""

from typing import Any

import pytest
from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch):
    """Ensure test runs start with controlled environment variables."""
    # Clear any environment variables that might interfere with tests
    for var in [
        "APP_NAME",
        "APP_ENV",
        "LOG_LEVEL",
        "MAX_UPLOAD_SIZE_MB",
        "AGENT_MAX_ITERATIONS",
        "AGENT_EXECUTION_TIMEOUT_SECONDS",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)

    # Clear cached get_settings singleton
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_session_state() -> dict[str, Any]:
    """Provide a clean dictionary mimicking Streamlit's session_state for isolated tests."""
    return {}
