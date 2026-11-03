from pathlib import Path

from pydantic import Field, SecretStr, model_validator
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
    database_url: SecretStr = SecretStr(
        "postgresql://sec_research:sec_research@127.0.0.1:5432/sec_research"
    )
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = Field(default="qwen3:8b", min_length=1)
    chat_timeout_seconds: float = Field(default=180, gt=0, le=600)
    chat_context_size: int = Field(default=16384, ge=2048, le=32768)
    chat_max_tokens: int = Field(default=2048, ge=128, le=8192)
    research_max_tool_calls: int = Field(default=6, ge=1, le=12)
    research_max_context_chars: int = Field(default=24000, ge=4000, le=48000)
    embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    embedding_dimensions: int = Field(default=768, ge=1, le=4096)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    embedding_timeout_seconds: float = Field(default=120, gt=0, le=600)
    rag_chunk_size: int = Field(default=1800, ge=100, le=6000)
    rag_chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def valid_chunk_overlap(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size")
        return self
