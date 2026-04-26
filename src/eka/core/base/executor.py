from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from eka.core.types import ExecutionResult, TraceEvent


class BaseExecutor(ABC):
    """Executor abstraction for orchestrating an agent workflow."""

    @abstractmethod
    def invoke(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        raise NotImplementedError

    def stream(self, user_input: str, session_id: str = "default") -> Iterator[TraceEvent]:
        """Yield observable execution events.

        Executors that support native graph/event streaming should override this method.
        """
        result = self.invoke(user_input=user_input, session_id=session_id)
        yield from result.trace

