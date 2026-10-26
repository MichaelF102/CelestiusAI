"""Application configuration management with Pydantic validation and security controls."""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env file if available, but do not fail if missing
load_dotenv(override=False)

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseModel):
    """Application settings schema with validation and sensitive field masking."""

    app_name: str = Field(default="CelestiusAI", min_length=1)
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development")
    log_level: LogLevelType = Field(default="INFO")

    max_upload_size_mb: int = Field(default=100, ge=1, le=2048)
    agent_max_iterations: int = Field(default=10, ge=1, le=100)
    agent_execution_timeout_seconds: int = Field(default=120, ge=5, le=3600)

    # Optional LLM integration (Gemini API)
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-pro")

    # Optional additional providers
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-5-5")
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1")

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str | None) -> str:
        if v is None:
            return "INFO"
        v_upper = str(v).strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v_upper not in allowed:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of {sorted(allowed)}")
        return v_upper

    @property
    def is_llm_configured(self) -> bool:
        """Safe boolean check indicating whether an LLM API key has been supplied."""
        return bool(
            (self.gemini_api_key and self.gemini_api_key.strip())
            or (self.anthropic_api_key and self.anthropic_api_key.strip())
        )

    @property
    def masked_gemini_api_key(self) -> str:
        """Returns masked API key representation without exposing secrets."""
        if not self.is_llm_configured:
            return "Not configured"
        key = self.gemini_api_key.strip()  # type: ignore[union-attr]
        if len(key) <= 8:
            return "********"
        return f"{key[:4]}...{key[-4:]}"

    def safe_dict(self) -> dict[str, object]:
        """Dictionary representation safe for UI display and non-sensitive logging."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env,
            "log_level": self.log_level,
            "max_upload_size_mb": self.max_upload_size_mb,
            "agent_max_iterations": self.agent_max_iterations,
            "agent_execution_timeout_seconds": self.agent_execution_timeout_seconds,
            "is_llm_configured": self.is_llm_configured,
            "masked_gemini_api_key": self.masked_gemini_api_key,
            "gemini_model": self.gemini_model,
        }

    def __repr__(self) -> str:
        return (
            f"AppSettings(app_name='{self.app_name}', app_version='{self.app_version}', "
            f"app_env='{self.app_env}', log_level='{self.log_level}', "
            f"max_upload_size_mb={self.max_upload_size_mb}, "
            f"agent_max_iterations={self.agent_max_iterations}, "
            f"agent_execution_timeout_seconds={self.agent_execution_timeout_seconds}, "
            f"is_llm_configured={self.is_llm_configured})"
        )


def load_settings_from_env() -> AppSettings:
    """Instantiate AppSettings from environment variables with sensible fallbacks."""
    raw_upload = os.getenv("MAX_UPLOAD_SIZE_MB", "100")
    try:
        max_upload = int(raw_upload)
    except ValueError as err:
        raise ValueError(f"MAX_UPLOAD_SIZE_MB must be an integer, got '{raw_upload}'") from err

    raw_iterations = os.getenv("AGENT_MAX_ITERATIONS", "10")
    try:
        max_iterations = int(raw_iterations)
    except ValueError as err:
        raise ValueError(
            f"AGENT_MAX_ITERATIONS must be an integer, got '{raw_iterations}'"
        ) from err

    raw_timeout = os.getenv("AGENT_EXECUTION_TIMEOUT_SECONDS", "120")
    try:
        execution_timeout = int(raw_timeout)
    except ValueError as err:
        raise ValueError(
            f"AGENT_EXECUTION_TIMEOUT_SECONDS must be an integer, got '{raw_timeout}'"
        ) from err

    return AppSettings(
        app_name=os.getenv("APP_NAME", "CelestiusAI"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
        max_upload_size_mb=max_upload,
        agent_max_iterations=max_iterations,
        agent_execution_timeout_seconds=execution_timeout,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached singleton provider for application settings."""
    return load_settings_from_env()
