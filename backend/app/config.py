from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEC_RESEARCH_",
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="SEC Research Desk", min_length=1)
    debug: bool = False
    sec_user_agent: str | None = Field(
        default=None,
        min_length=5,
        pattern=r"^[^\r\n]+ [^\s@]+@[^\s@]+\.[^\s@]+$",
    )
    sec_cache_dir: Path = Path(__file__).resolve().parents[2] / ".cache" / "sec"
    sec_cache_ttl_seconds: int = Field(default=3600, ge=0)
    sec_request_interval_seconds: float = Field(default=0.2, ge=0.1, le=60)
    sec_timeout_seconds: float = Field(default=30, gt=0, le=120)
    sec_max_retries: int = Field(default=2, ge=0, le=5)
