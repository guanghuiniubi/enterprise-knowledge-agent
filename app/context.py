from __future__ import annotations

from contextvars import ContextVar

_request_context: ContextVar[dict] = ContextVar("request_context", default={})


def set_request_context(**kwargs):
    current = dict(_request_context.get() or {})
    current.update({key: value for key, value in kwargs.items() if value is not None})
    _request_context.set(current)


def get_request_context() -> dict:
    return dict(_request_context.get() or {})


def clear_request_context():
    _request_context.set({})
