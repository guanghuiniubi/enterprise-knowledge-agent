from __future__ import annotations

import os
from typing import Any, cast

from langchain_openai import ChatOpenAI

from eka.config.settings import Settings, get_settings

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "xiaomi", "openai_compatible", "compatible"}


def configure_observability(settings: Settings) -> None:
    """Populate standard LangSmith environment variables when enabled."""

    if not settings.langsmith_tracing:
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)


def build_chat_model(settings: Settings | None = None, **overrides: Any) -> ChatOpenAI:
    if settings is None:
        settings = get_settings()
    settings = cast(Settings, settings)
    configure_observability(settings)

    provider = settings.llm_provider.lower().strip()
    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{settings.llm_provider}'. "
            f"Currently supported: {', '.join(sorted(OPENAI_COMPATIBLE_PROVIDERS))}."
        )

    if not settings.llm_api_key:
        raise ValueError("Missing LLM_API_KEY in environment configuration.")

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": settings.llm_temperature,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)

