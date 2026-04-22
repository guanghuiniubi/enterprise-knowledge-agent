from __future__ import annotations

from abc import ABC, abstractmethod

from eka.core.types import MessageRecord


class BaseMemory(ABC):
    """Conversation memory abstraction."""

    @abstractmethod
    def load(self, session_id: str) -> list[MessageRecord]:
        raise NotImplementedError

    @abstractmethod
    def save_turn(self, session_id: str, user_input: str, assistant_output: str) -> None:
        raise NotImplementedError

