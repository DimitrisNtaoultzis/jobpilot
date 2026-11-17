import pytest
from langchain_groq import ChatGroq
from pydantic import ValidationError

from jobpilot.ai_service.config import Settings
from jobpilot.ai_service.llm import get_chat_model

# Settings(_env_file=None): read only the variables set by the test, never the local .env.
# GROQ_API_KEY is set to a fake value by the autouse fixture in conftest.py.

FAKE_DATABASE_URL = "postgresql://user:not-real-password@localhost:5432/db"


@pytest.fixture(autouse=True)
def fake_database_url(monkeypatch):
    """Settings require DATABASE_URL; set a fake one for this module only.

    Not in conftest.py: database tests elsewhere need the real URL.
    """
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)


def test_settings_read_the_key_and_default_model():
    settings = Settings(_env_file=None)

    assert settings.groq_api_key.get_secret_value() == "test-key-not-real"
    assert settings.groq_model == "openai/gpt-oss-120b"


def test_settings_require_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_an_empty_model_setting_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "")

    assert Settings(_env_file=None).groq_model == "openai/gpt-oss-120b"


def test_the_api_key_is_hidden_in_repr():
    assert "test-key-not-real" not in repr(Settings(_env_file=None))


def test_settings_read_the_database_url():
    settings = Settings(_env_file=None)

    assert settings.database_url.get_secret_value() == FAKE_DATABASE_URL


def test_settings_require_a_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_the_database_password_is_hidden_in_repr():
    assert "not-real-password" not in repr(Settings(_env_file=None))


def test_get_chat_model_uses_the_settings(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    model = get_chat_model()  # building the model makes no network call; .invoke() would

    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-20b"
