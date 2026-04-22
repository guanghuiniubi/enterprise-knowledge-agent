from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from eka.core.types import ExecutionResult, PlanRecord, TraceEvent


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

