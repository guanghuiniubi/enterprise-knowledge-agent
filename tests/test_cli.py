from __future__ import annotations

from eka.__main__ import render_answer
from eka.core.types import ExecutionResult, PlanRecord, RetrievedDocument, TraceEvent


def test_render_answer_contains_plan_docs_answer_and_trace():
    result = ExecutionResult(
        session_id="cli",
        answer="这是最终回答",
        plan=PlanRecord(
            objective="准备 LangGraph 面试",
            reasoning_summary="先讲状态图，再讲工具和记忆。",
            search_queries=["LangGraph interview"],
            tools_to_consider=["interview_checklist"],
        ),
        retrieved_docs=[
            RetrievedDocument(
                source="/tmp/demo.md",
                content="LangGraph supports stateful orchestration.",
                score=0.9,
                metadata={"filename": "demo.md"},
            )
        ],
        trace=[TraceEvent(stage="assistant", message="生成最终回答")],
    )

    rendered = render_answer(result)

    assert "[Plan Summary]" in rendered
    assert "先讲状态图" in rendered
    assert "demo.md" in rendered
    assert "这是最终回答" in rendered
    assert "[Trace]" in rendered
    assert "assistant: 生成最终回答" in rendered

