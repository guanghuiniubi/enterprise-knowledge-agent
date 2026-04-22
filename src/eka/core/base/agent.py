from __future__ import annotations

from abc import ABC, abstractmethod

from eka.core.types import ExecutionResult


class BaseAgent(ABC):
    """Top-level agent abstraction."""

    name: str
    description: str

    @abstractmethod
    def respond(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        raise NotImplementedError

