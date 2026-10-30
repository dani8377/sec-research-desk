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
    [("SEC_RESEARCH_DEBUG", "invalid"), ("SEC_RESEARCH_APP_NAME", "")],
)
def test_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
