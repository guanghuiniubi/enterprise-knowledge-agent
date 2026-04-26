from __future__ import annotations

import os

import pytest

import eka.config.providers as providers
from eka.config.providers import build_chat_model, configure_observability, resolve_base_url
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


def test_resolve_base_url_uses_xiaomi_default_when_not_configured():
    settings = Settings(
        APP_NAME="demo",
        LLM_PROVIDER="xiaomi",
        LLM_MODEL="mimo-v2-pro",
        LLM_API_KEY="secret",
    )

    assert resolve_base_url(settings) == "https://api.xiaomimimo.com/v1"


def test_build_chat_model_passes_openai_compatible_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(providers, "ChatOpenAI", DummyChatOpenAI)

    settings = Settings(
        APP_NAME="demo",
        LLM_PROVIDER="xiaomi",
        LLM_MODEL="mimo-v2-pro",
        LLM_API_KEY="secret",
        LLM_TEMPERATURE="0.3",
        LLM_TIMEOUT="45",
        LLM_MAX_RETRIES="4",
        LLM_MAX_TOKENS="2048",
    )

    build_chat_model(settings)

    assert captured["model"] == "mimo-v2-pro"
    assert captured["api_key"] == "secret"
    assert captured["temperature"] == 0.3
    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == 4
    assert captured["max_tokens"] == 2048
    assert captured["base_url"] == "https://api.xiaomimimo.com/v1"


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
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-secret"
    assert os.environ["LANGSMITH_PROJECT"] == "eka-test"

