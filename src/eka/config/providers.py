from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from eka.config.settings import Settings, get_settings

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "xiaomi", "openai_compatible", "compatible"}
DEFAULT_PROVIDER_BASE_URLS = {
    "xiaomi": "https://api.xiaomimimo.com/v1",
}


def configure_observability(settings: Settings) -> None:
    """Populate standard LangSmith environment variables when enabled."""

    if not settings.langsmith_tracing:
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)


def resolve_base_url(settings: Settings) -> str | None:
    if settings.llm_base_url:
        return settings.llm_base_url
    return DEFAULT_PROVIDER_BASE_URLS.get(settings.normalized_llm_provider)


def build_chat_model(settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    resolved_settings = cast(Settings, settings or get_settings())
    configure_observability(resolved_settings)

    provider = resolved_settings.normalized_llm_provider
    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{resolved_settings.llm_provider}'. "
            f"Currently supported: {', '.join(sorted(OPENAI_COMPATIBLE_PROVIDERS))}."
        )

    if not resolved_settings.llm_api_key:
        raise ValueError("Missing LLM_API_KEY in environment configuration.")
    if not resolved_settings.llm_model:
        raise ValueError("Missing LLM_MODEL in environment configuration.")

    kwargs: dict[str, Any] = {
        "model": resolved_settings.llm_model,
        "api_key": resolved_settings.llm_api_key,
        "temperature": resolved_settings.llm_temperature,
        "timeout": resolved_settings.llm_timeout,
        "max_retries": resolved_settings.llm_max_retries,
    }
    base_url = resolve_base_url(resolved_settings)
    if base_url:
        kwargs["base_url"] = base_url
    if resolved_settings.llm_max_tokens is not None:
        kwargs["max_tokens"] = resolved_settings.llm_max_tokens
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)

