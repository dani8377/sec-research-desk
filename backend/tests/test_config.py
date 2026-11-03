from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_RESEARCH_APP_NAME", raising=False)
    monkeypatch.delenv("SEC_RESEARCH_DEBUG", raising=False)


def test_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "SEC Research Desk"
    assert settings.debug is False
    assert settings.sec_user_agent is None
    assert settings.sec_request_interval_seconds == 0.2
    assert settings.sec_cache_ttl_seconds == 3600


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_RESEARCH_APP_NAME", "Research Test")
    monkeypatch.setenv("SEC_RESEARCH_DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Research Test"
    assert settings.debug is True


def test_dotenv_and_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SEC_RESEARCH_APP_NAME=From dotenv\nSEC_RESEARCH_DEBUG=true\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.app_name == "From dotenv"
    assert settings.debug is True

    monkeypatch.setenv("SEC_RESEARCH_APP_NAME", "From environment")
    assert Settings(_env_file=env_file).app_name == "From environment"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SEC_RESEARCH_DEBUG", "invalid"),
        ("SEC_RESEARCH_APP_NAME", ""),
        ("SEC_RESEARCH_SEC_USER_AGENT", "No contact"),
        ("SEC_RESEARCH_SEC_USER_AGENT", "Research\r\ntest@example.com"),
        ("SEC_RESEARCH_SEC_CACHE_TTL_SECONDS", "-1"),
        ("SEC_RESEARCH_SEC_REQUEST_INTERVAL_SECONDS", "0.01"),
        ("SEC_RESEARCH_SEC_TIMEOUT_SECONDS", "0"),
        ("SEC_RESEARCH_SEC_MAX_RETRIES", "6"),
        ("SEC_RESEARCH_EMBEDDING_DIMENSIONS", "0"),
        ("SEC_RESEARCH_EMBEDDING_BATCH_SIZE", "0"),
        ("SEC_RESEARCH_EMBEDDING_TIMEOUT_SECONDS", "0"),
        ("SEC_RESEARCH_RAG_CHUNK_SIZE", "99"),
        ("SEC_RESEARCH_RAG_CHUNK_OVERLAP", "1800"),
        ("SEC_RESEARCH_CHAT_MODEL", ""),
        ("SEC_RESEARCH_CHAT_TIMEOUT_SECONDS", "0"),
        ("SEC_RESEARCH_CHAT_CONTEXT_SIZE", "0"),
        ("SEC_RESEARCH_CHAT_MAX_TOKENS", "0"),
        ("SEC_RESEARCH_RESEARCH_MAX_TOOL_CALLS", "0"),
        ("SEC_RESEARCH_RESEARCH_MAX_CONTEXT_CHARS", "0"),
    ],
)
def test_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
