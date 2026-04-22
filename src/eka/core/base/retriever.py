from __future__ import annotations

from abc import ABC, abstractmethod

from eka.core.types import RetrievedDocument


class BaseRetriever(ABC):
    """Knowledge retriever abstraction."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 4) -> list[RetrievedDocument]:
        raise NotImplementedError

