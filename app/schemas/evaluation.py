from typing import Any

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    max_latency_ms: float | None = None
    expect_fallback: bool | None = None


class EvaluationRunRequest(BaseModel):
    cases: list[EvaluationCase]


class EvaluationCaseResult(BaseModel):
    case_id: str
    question: str
    answer: str
    accuracy: float
    step_count: int
    latency_ms: float
    fallback: bool
    passed: bool
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    violated_keywords: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    total_cases: int
    correct_cases: int
    accuracy: float
    avg_step_count: float
    avg_latency_ms: float
    fallback_rate: float


class EvaluationRunResponse(BaseModel):
    summary: EvaluationSummary
    results: list[EvaluationCaseResult]

