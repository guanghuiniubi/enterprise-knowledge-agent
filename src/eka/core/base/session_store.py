from __future__ import annotations

from abc import ABC, abstractmethod

from eka.core.types import MessageRecord


class BaseSessionStore(ABC):
    """Persistence abstraction for session conversations."""

    @abstractmethod
    def get_messages(self, session_id: str) -> list[MessageRecord]:
        raise NotImplementedError

    @abstractmethod
    def append_messages(self, session_id: str, messages: list[MessageRecord]) -> None:
        raise NotImplementedError

