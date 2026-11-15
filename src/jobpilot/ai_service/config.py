"""AI service settings (pydantic-settings): the counterpart of django-environ for this service."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (this file is at src/jobpilot/ai_service/config.py), as in the Django settings
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """AI service settings, read from environment variables or the .env file.

    Field names map to variable names case-insensitively (groq_api_key <- GROQ_API_KEY).
    Real environment variables take precedence over the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        # The shared .env also holds Django settings (SECRET_KEY, ...)
        extra="ignore",
        # Treat "GROQ_MODEL=" (as in .env.example) as unset, so the default is used
        env_ignore_empty=True,
    )

    # SecretStr hides the value in repr/logs; read it with .get_secret_value().
    # No default: a missing key fails right away with a clear error.
    groq_api_key: SecretStr
    # A setting, not hard-coded: Groq's model catalog changes often.
    groq_model: str = "openai/gpt-oss-120b"


@lru_cache
def get_settings() -> Settings:
    """Load the settings once, on first use (not at import time).

    Loading lazily means importing the app (e.g. for the /health tests in CI)
    does not require a Groq API key.
    """
    return Settings()
