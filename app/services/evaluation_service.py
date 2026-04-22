from __future__ import annotations

import time
from collections.abc import Callable

from app.agent.orchestrator import agent_orchestrator
from app.core.config import settings
from app.observability.metrics import observability_manager
from app.schemas.agent import AgentResult
from app.schemas.evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRunResponse,
    EvaluationSummary,
)


class EvaluationService:
    def __init__(self, runner: Callable[[str], AgentResult] | None = None):
        self._runner = runner or (lambda question: agent_orchestrator.run(question))

    def _keyword_metrics(self, answer: str, case: EvaluationCase) -> tuple[list[str], list[str], list[str], float]:
        answer_lower = answer.lower()
        expected = [item for item in case.expected_keywords if item]
        matched = [item for item in expected if item.lower() in answer_lower]
        missing = [item for item in expected if item.lower() not in answer_lower]
        violated = [item for item in case.forbidden_keywords if item.lower() in answer_lower]
        if expected:
            accuracy = len(matched) / len(expected)
        else:
            accuracy = 1.0 if answer.strip() else 0.0
        if violated:
            accuracy = max(0.0, accuracy - 0.2 * len(violated))
        return matched, missing, violated, round(accuracy, 4)

    def _is_passed(
        self,
        *,
        accuracy: float,
        step_count: int,
        latency_ms: float,
        fallback: bool,
        violated_keywords: list[str],
        case: EvaluationCase,
    ) -> bool:
        if violated_keywords:
            return False
        if accuracy < settings.evaluation_pass_accuracy_threshold:
            return False
        if case.max_steps is not None and step_count > case.max_steps:
            return False
        if case.max_latency_ms is not None and latency_ms > case.max_latency_ms:
            return False
        if case.expect_fallback is not None and fallback != case.expect_fallback:
            return False
        return True

    def run_cases(self, cases: list[EvaluationCase]) -> EvaluationRunResponse:
        results: list[EvaluationCaseResult] = []

        for case in cases:
            started = time.perf_counter()
            agent_result = self._runner(case.question)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            step_count = len(agent_result.agent_steps)
            fallback = bool((agent_result.debug or {}).get("fallback"))
            matched, missing, violated, accuracy = self._keyword_metrics(agent_result.answer, case)
            passed = self._is_passed(
                accuracy=accuracy,
                step_count=step_count,
                latency_ms=latency_ms,
                fallback=fallback,
                violated_keywords=violated,
                case=case,
            )
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer=agent_result.answer,
                    accuracy=accuracy,
                    step_count=step_count,
                    latency_ms=latency_ms,
                    fallback=fallback,
                    passed=passed,
                    matched_keywords=matched,
                    missing_keywords=missing,
                    violated_keywords=violated,
                    debug=agent_result.debug or {},
                )
            )

        total = len(results)
        correct_cases = sum(1 for item in results if item.passed)
        summary = EvaluationSummary(
            total_cases=total,
            correct_cases=correct_cases,
            accuracy=round(correct_cases / total, 4) if total else 0.0,
            avg_step_count=round(sum(item.step_count for item in results) / total, 4) if total else 0.0,
            avg_latency_ms=round(sum(item.latency_ms for item in results) / total, 4) if total else 0.0,
            fallback_rate=round(sum(1 for item in results if item.fallback) / total, 4) if total else 0.0,
        )
        observability_manager.record_evaluation_run(
            case_count=total,
            accuracy=summary.accuracy,
            fallback_rate=summary.fallback_rate,
        )
        return EvaluationRunResponse(summary=summary, results=results)


# Default singleton for API usage.
evaluation_service = EvaluationService()

