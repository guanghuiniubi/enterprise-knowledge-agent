import time
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.core.governance import CircuitBreakerOpen, ExecutionTimeout, GovernanceManager
from app.prompts.registry import PromptRegistry
from app.rag.reranker import HybridReranker
from app.schemas.agent import AgentResult
from app.schemas.evaluation import EvaluationCase
from app.services.evaluation_service import EvaluationService


def test_prompt_registry_supports_version_activation():
    registry = PromptRegistry()
    registry.register(
        name="agent_system",
        version="v2",
        template="system={mode}",
        description="new version",
    )

    activated = registry.activate("agent_system", "v2")

    assert activated.version == "v2"
    assert registry.render("agent_system", mode="strict") == "system=strict"
    assert registry.active_versions()["agent_system"] == "v2"


def test_governance_opens_circuit_after_repeated_timeouts(monkeypatch):
    manager = GovernanceManager()
    monkeypatch.setattr(settings, "tool_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "tool_circuit_failure_threshold", 2)
    monkeypatch.setattr(settings, "tool_circuit_recovery_seconds", 60.0)
    monkeypatch.setattr(settings, "tool_rate_limit_max_requests", 100)

    with pytest.raises(ExecutionTimeout):
        manager.execute_tool("slow_tool", lambda: time.sleep(0.05))

    with pytest.raises(ExecutionTimeout):
        manager.execute_tool("slow_tool", lambda: time.sleep(0.05))

    with pytest.raises(CircuitBreakerOpen):
        manager.execute_tool("slow_tool", lambda: "never called")


def test_hybrid_reranker_prefers_better_title_match():
    reranker = HybridReranker()
    rows = [
        {
            "id": "1",
            "document_id": "10",
            "title": "Python 垃圾回收机制",
            "content": "Python 通过引用计数、标记清除和分代回收处理对象生命周期。",
            "score": 0.55,
            "metadata": {"tags": ["Python", "GC"]},
        },
        {
            "id": "2",
            "document_id": "11",
            "title": "缓存一致性",
            "content": "这篇内容顺带提到了 Python，但是并不聚焦垃圾回收。",
            "score": 0.72,
            "metadata": {"tags": ["Cache"]},
        },
    ]

    ranked = reranker.rerank_vector_results("Python 垃圾回收", rows, top_k=2)

    assert ranked[0]["id"] == "1"
    assert ranked[0]["rerank_score"] >= ranked[1]["rerank_score"]


def test_hybrid_reranker_supports_chunk_reranking():
    reranker = HybridReranker()
    chunks = [
        SimpleNamespace(chunk_index=1, header_path="背景", content="介绍缓存一致性", metadata_json={}),
        SimpleNamespace(chunk_index=0, header_path="Python 垃圾回收", content="引用计数和分代回收", metadata_json={"tags": ["Python"]}),
    ]

    ranked = reranker.rerank_chunks("Python 垃圾回收", chunks)

    assert ranked[0].header_path == "Python 垃圾回收"


def test_evaluation_service_aggregates_accuracy_steps_latency_and_fallback():
    responses = {
        "q1": AgentResult(
            answer="核心结论：TCP 三次握手用于确认双方收发能力和初始序列号。",
            route="agent_answer",
            agent_steps=[{"step": 1}, {"step": 2}],
            debug={"fallback": False},
        ),
        "q2": AgentResult(
            answer="系统降级了，但仍然给出保守回答。",
            route="agent_answer",
            agent_steps=[{"step": 1}],
            debug={"fallback": True},
        ),
    }
    service = EvaluationService(runner=lambda question: responses[question])

    report = service.run_cases([
        EvaluationCase(case_id="c1", question="q1", expected_keywords=["三次握手", "序列号"], max_steps=3),
        EvaluationCase(case_id="c2", question="q2", expected_keywords=["降级"], expect_fallback=True),
    ])

    assert report.summary.total_cases == 2
    assert report.summary.correct_cases == 2
    assert report.summary.accuracy == 1.0
    assert report.summary.avg_step_count == 1.5
    assert report.summary.fallback_rate == 0.5
    assert report.results[1].fallback is True

