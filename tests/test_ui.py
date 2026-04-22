from fastapi.testclient import TestClient

from app.main import app
from app.memory.session_store import session_store
from app.observability.metrics import observability_manager
from app.schemas.agent import AgentResult
from app.services.evaluation_service import evaluation_service


client = TestClient(app)


def test_root_ui_page_loads():
    response = client.get("/")

    assert response.status_code == 200
    assert "Knowledge Agent UI" in response.text
    assert "/chat/stream" in response.text
    assert "observability dashboard" in response.text.lower()
    assert "Derived Metrics" not in response.text
    assert "Latency Observations" not in response.text
    assert "Governance Snapshot" not in response.text
    assert "Agent Steps" not in response.text
    assert '<pre class="json">' not in response.text
    assert 'class="messages"' in response.text
    assert 'class="composer"' in response.text
    assert "position: sticky;" in response.text


def test_sessions_api_returns_persisted_messages():
    session_id = "ui-test-session"
    session_store.clear(session_id)
    session_store.add_message(session_id, "user", "hello", user_id="tester")
    session_store.add_message(session_id, "assistant", "world", user_id="tester")

    sessions_response = client.get("/sessions")
    messages_response = client.get(f"/sessions/{session_id}/messages")

    assert sessions_response.status_code == 200
    assert any(item["session_id"] == session_id for item in sessions_response.json()["sessions"])
    assert messages_response.status_code == 200
    assert len(messages_response.json()["messages"]) >= 2


def test_prompt_registry_api_exposes_active_versions():
    response = client.get("/prompts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_versions"]["agent_system"] == "v1"
    assert any(item["name"] == "agent_system" for item in payload["prompts"])


def test_evaluation_api_returns_accuracy_and_fallback_metrics(monkeypatch):
    monkeypatch.setattr(
        evaluation_service,
        "_runner",
        lambda question: AgentResult(
            answer="核心结论：Redis 持久化包括 RDB 和 AOF。",
            route="agent_answer",
            agent_steps=[{"step": 1}, {"step": 2}],
            debug={"fallback": False},
        ),
    )

    response = client.post(
        "/evaluation/run",
        json={
            "cases": [
                {
                    "case_id": "redis-1",
                    "question": "Redis 持久化有哪些方式？",
                    "expected_keywords": ["RDB", "AOF"],
                    "max_steps": 3,
                    "expect_fallback": False,
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["accuracy"] == 1.0
    assert payload["summary"]["fallback_rate"] == 0.0
    assert payload["results"][0]["passed"] is True


def test_governance_api_returns_runtime_snapshot():
    response = client.get("/governance")

    assert response.status_code == 200
    payload = response.json()
    assert "llm" in payload
    assert "tool" in payload


def test_observability_overview_returns_dashboard_payload():
    observability_manager.reset()
    observability_manager.record_chat_request(
        route="agent_answer",
        latency_ms=123.0,
        fallback=True,
        step_count=2,
        tool_calls=2,
        tool_failures=1,
    )
    observability_manager.record_prompt_injection_check(hit=True, blocked=True)
    observability_manager.record_acl_check(allowed=False, stage="vector_search", visibility="restricted", reason="insufficient_clearance")

    response = client.get("/observability/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["kpis"]
    assert payload["derived"]["fallback_rate"] == 1.0
    assert payload["derived"]["prompt_injection_hit_rate"] == 1.0
    assert payload["derived"]["acl_deny_rate"] == 1.0
    assert "governance" in payload
    assert "alerts" in payload


