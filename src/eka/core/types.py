from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageRecord:
    role: str
    content: str


@dataclass(slots=True)
class PlanRecord:
    objective: str
    reasoning_summary: str
    search_queries: list[str] = field(default_factory=list)
    tools_to_consider: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedDocument:
    source: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str


@dataclass(slots=True)
class TraceEvent:
    stage: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    session_id: str
    answer: str
    plan: PlanRecord | None = None
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)

