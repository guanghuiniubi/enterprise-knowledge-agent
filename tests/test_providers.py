from __future__ import annotations

import os

import pytest

from eka.config.providers import build_chat_model, configure_observability
from eka.config.settings import Settings


def test_build_chat_model_rejects_unsupported_provider():
    settings = Settings(
        APP_NAME="demo",
        LLM_PROVIDER="anthropic",
        LLM_MODEL="dummy",
        LLM_API_KEY="secret",
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_chat_model(settings)


def test_build_chat_model_requires_api_key():
    settings = Settings(
        APP_NAME="demo",
        LLM_PROVIDER="xiaomi",
        LLM_MODEL="dummy",
        LLM_API_KEY="",
    )

    with pytest.raises(ValueError, match="Missing LLM_API_KEY"):
        build_chat_model(settings)


def test_configure_observability_sets_langsmith_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    settings = Settings(
        APP_NAME="demo",
        LANGSMITH_TRACING="true",
        LANGSMITH_API_KEY="ls-secret",
        LANGSMITH_PROJECT="eka-test",
    )

    configure_observability(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-secret"
    assert os.environ["LANGSMITH_PROJECT"] == "eka-test"

