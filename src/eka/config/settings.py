from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="enterprise-knowledge-agent", validation_alias="APP_NAME")
    app_env: str = Field(default="dev", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    app_api_key: str | None = Field(default=None, validation_alias="APP_API_KEY")

    llm_provider: str = Field(default="xiaomi", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="mimo-v2-pro", validation_alias="LLM_MODEL")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_temperature: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")

    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="enterprise-knowledge-agent", validation_alias="LANGSMITH_PROJECT")

    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ai_rag_db", validation_alias="POSTGRES_DB")
    postgres_user: str | None = Field(default=None, validation_alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, validation_alias="POSTGRES_PASSWORD")

    knowledge_base_path: Path | None = Field(default=None, validation_alias="KNOWLEDGE_BASE_PATH")
    embedding_model_name: str = Field(default="BAAI/bge-small-zh-v1.5", validation_alias="EMBEDDING_MODEL_NAME")
    embedding_dimension: int = Field(default=512, validation_alias="EMBEDDING_DIMENSION")

    @property
    def postgres_dsn(self) -> str:
        user = self.postgres_user or ""
        password = self.postgres_password or ""
        auth = f"{user}:{password}@" if user or password else ""
        return f"postgresql+psycopg://{auth}{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def knowledge_base_dir(self) -> Path | None:
        return self.knowledge_base_path.expanduser().resolve() if self.knowledge_base_path else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

