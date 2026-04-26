from __future__ import annotations

from eka.__main__ import render_answer
from eka.core.types import (
    ExecutionResult,
    PlanCandidateRecord,
    PlanRecord,
    RetrievedDocument,
    RouteTraceRecord,
    ToolCallRecord,
    TraceEvent,
)


def test_render_answer_contains_plan_docs_answer_and_trace():
    result = ExecutionResult(
        session_id="cli",
        answer="这是最终回答",
        plan=PlanRecord(
            objective="准备 LangGraph 面试",
            reasoning_summary="先讲状态图，再讲工具和记忆。",
            search_queries=["LangGraph interview"],
            tools_to_consider=["interview_checklist"],
            template_id="general_interview",
            candidate_template_ids=["general_interview", "preparation_checklist"],
            candidate_details=[
                PlanCandidateRecord(
                    template_id="general_interview",
                    score=2,
                    priority=10,
                    matched_keywords=["langgraph", "面试"],
                    selected=True,
                ),
                PlanCandidateRecord(
                    template_id="preparation_checklist",
                    score=1,
                    priority=20,
                    matched_keywords=["准备"],
                    selected=False,
                    rejected_reason="Not selected by LLM rerank; 'general_interview' ranked higher.",
                ),
            ],
            selection_strategy="rule_plus_llm_rerank",
            selection_reason="The request is broad and best handled by the general interview template.",
            selection_confidence=0.88,
            route_trace=[
                RouteTraceRecord(stage="rule_recall", message="Recalled 2 candidate templates."),
                RouteTraceRecord(stage="llm_rerank", message="LLM rerank selected template 'general_interview'."),
            ],
        ),
        retrieved_docs=[
            RetrievedDocument(
                source="/tmp/demo.md",
                content="LangGraph supports stateful orchestration.",
                score=0.9,
                metadata={"filename": "demo.md"},
            )
        ],
        tool_calls=[
            ToolCallRecord(
                tool_name="interview_checklist",
                tool_input={"topic": "LangGraph"},
                tool_output="1. 讲清状态图",
            )
        ],
        trace=[TraceEvent(stage="assistant", message="生成最终回答")],
    )

    rendered = render_answer(result)

    assert "[Plan Summary]" in rendered
    assert "先讲状态图" in rendered
    assert "[Plan Route]" in rendered
    assert "Template: general_interview" in rendered
    assert "Candidates: general_interview, preparation_checklist" in rendered
    assert "[Plan Candidates]" in rendered
    assert "general_interview: score=2" in rendered
    assert "preparation_checklist: score=1" in rendered
    assert "[Plan Route Trace]" in rendered
    assert "rule_recall: Recalled 2 candidate templates." in rendered
    assert "demo.md" in rendered
    assert "[Tool Calls]" in rendered
    assert "interview_checklist" in rendered
    assert "这是最终回答" in rendered
    assert "[Trace]" in rendered
    assert "assistant: 生成最终回答" in rendered

