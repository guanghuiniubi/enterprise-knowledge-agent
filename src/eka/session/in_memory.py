from __future__ import annotations

from collections import defaultdict

from eka.core.base import BaseSessionStore
from eka.core.types import MessageRecord


class InMemorySessionStore(BaseSessionStore):
    """Simple in-memory store for multi-turn conversations."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[MessageRecord]] = defaultdict(list)

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        return list(self._sessions.get(session_id, []))

    def append_messages(self, session_id: str, messages: list[MessageRecord]) -> None:
        self._sessions[session_id].extend(messages)

