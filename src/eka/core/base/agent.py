from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from eka.core.types import ExecutionResult, TraceEvent


class BaseAgent(ABC):
    """Top-level agent abstraction."""

    name: str
    description: str

    @abstractmethod
    def respond(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        raise NotImplementedError

    def stream(self, user_input: str, session_id: str = "default") -> Iterator[TraceEvent]:
        result = self.respond(user_input=user_input, session_id=session_id)
        yield from result.trace

