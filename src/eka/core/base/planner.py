from __future__ import annotations

from abc import ABC, abstractmethod

from eka.core.types import MessageRecord, PlanRecord


class BasePlanner(ABC):
    """Planner abstraction for generating action summaries."""

    @abstractmethod
    def create_plan(self, user_input: str, history: list[MessageRecord]) -> PlanRecord:
        raise NotImplementedError

