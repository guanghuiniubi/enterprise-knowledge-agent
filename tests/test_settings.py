from __future__ import annotations

from eka.config.settings import get_settings


def test_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("APP_NAME", "eka-test")
    monkeypatch.setenv("LLM_PROVIDER", "xiaomi")
    monkeypatch.setenv("LLM_MODEL", "demo-model")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.app_name == "eka-test"
    assert settings.llm_model == "demo-model"
    assert settings.postgres_dsn == "postgresql+psycopg://tester:secret@localhost:5432/ai_rag_db"

    get_settings.cache_clear()

