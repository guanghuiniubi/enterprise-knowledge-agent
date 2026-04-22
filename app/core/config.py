from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "enterprise-knowledge-agent"
    app_env: str = "dev"
    log_level: str = "INFO"

    llm_provider: str = "xiaomi"
    llm_model: str = "mimo-v2-pro"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "dummy-key"

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "enterprise-knowledge-agent"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_rag_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    knowledge_base_path: str = "data/knowledge_docs.json"
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512

    request_rate_limit_window_seconds: int = 60
    request_rate_limit_max_requests: int = 20
    llm_rate_limit_window_seconds: int = 60
    llm_rate_limit_max_requests: int = 60
    tool_rate_limit_window_seconds: int = 60
    tool_rate_limit_max_requests: int = 120
    llm_timeout_seconds: float = 20.0
    tool_timeout_seconds: float = 8.0
    llm_circuit_failure_threshold: int = 3
    llm_circuit_recovery_seconds: float = 30.0
    tool_circuit_failure_threshold: int = 4
    tool_circuit_recovery_seconds: float = 15.0

    rerank_candidate_multiplier: int = 4
    rerank_weight_vector: float = 0.45
    rerank_weight_title: float = 0.2
    rerank_weight_keyword: float = 0.2
    rerank_weight_metadata: float = 0.1
    rerank_weight_position: float = 0.05
    rerank_diversity_lambda: float = 0.85

    evaluation_pass_accuracy_threshold: float = 1.0

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
