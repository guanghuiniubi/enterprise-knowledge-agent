from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "enterprise-knowledge-agent"
    app_env: str = "dev"
    log_level: str = "INFO"

    llm_provider: str = "xiaomi"
    llm_model: str = "mimo-v2-pro"
    llm_base_url: str
    llm_api_key: str

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "enterprise-knowledge-agent"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_rag_db"
    postgres_user: str
    postgres_password: str

    knowledge_base_path: str
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
