from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from eka.core.types import ExecutionResult, PlanCandidateRecord, PlanRecord, RouteTraceRecord, TraceEvent


app_module = importlib.import_module("eka.api.app")


class StubAgent:
	def respond(self, user_input: str, session_id: str = "default") -> ExecutionResult:
		return ExecutionResult(
			session_id=session_id,
			answer=f"stub:{user_input}",
			plan=PlanRecord(
				objective="demo",
				reasoning_summary="stub plan",
				search_queries=[user_input],
				tools_to_consider=["interview_checklist"],
				template_id="general_interview",
				candidate_template_ids=["general_interview"],
				selection_strategy="rule_based",
				selection_reason="matched keywords: 你好",
				selection_confidence=0.7,
				fallback_used=False,
				candidate_details=[
					PlanCandidateRecord(
						template_id="general_interview",
						score=2,
						priority=10,
						matched_keywords=["你好"],
						selected=True,
					)
				],
				route_trace=[
					RouteTraceRecord(stage="rule_recall", message="Recalled 1 candidate templates.")
				],
			),
			trace=[TraceEvent(stage="assistant", message="stub trace")],
		)


def test_chat_endpoint_returns_serialized_agent_result(monkeypatch):
	app_module.get_agent.cache_clear()
	monkeypatch.setattr(app_module, "get_agent", lambda: StubAgent())

	client = TestClient(app_module.create_app())
	response = client.post("/chat", json={"question": "你好", "session_id": "api"})

	assert response.status_code == 200
	payload = response.json()
	assert payload["answer"] == "stub:你好"
	assert payload["session_id"] == "api"
	assert payload["plan_summary"] == "stub plan"
	assert payload["plan_route"]["template_id"] == "general_interview"
	assert payload["plan_route"]["candidate_template_ids"] == ["general_interview"]
	assert payload["plan_route"]["selection_strategy"] == "rule_based"
	assert payload["plan_route"]["selection_reason"] == "matched keywords: 你好"
	assert payload["plan_route"]["selection_confidence"] == 0.7
	assert payload["plan_route"]["fallback_used"] is False
	assert payload["candidate_details"][0]["template_id"] == "general_interview"
	assert payload["candidate_details"][0]["selected"] is True
	assert payload["route_trace"][0]["stage"] == "rule_recall"

