from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_request_context: ContextVar[dict[str, Any]] = ContextVar("request_context", default={})


def set_request_context(**kwargs: Any):
    current = dict(_request_context.get() or {})
    current.update({key: value for key, value in kwargs.items() if value is not None})
    _request_context.set(current)


def get_request_context() -> dict[str, Any]:
    return dict(_request_context.get() or {})


def clear_request_context():
    _request_context.set({})

