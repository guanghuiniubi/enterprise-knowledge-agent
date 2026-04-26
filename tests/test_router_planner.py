from __future__ import annotations

from eka.core.types import MessageRecord
from eka.planner import RouterPlanner


class FakeStructuredRouterModel:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.invocations = []

    def with_structured_output(self, schema):
        parent = self

        class StructuredRunner:
            def invoke(self, messages):
                parent.invocations.append(messages)
                return schema(**parent.payload)

        return StructuredRunner()


def test_router_planner_uses_rule_based_template_when_keywords_match():
    planner = RouterPlanner.default()

    plan = planner.create_plan("请帮我把项目经历整理成 STAR 面试表达", history=[])

    assert plan.template_id == "star_story"
    assert plan.selection_strategy == "rule_based"
    assert plan.candidate_template_ids[0] == "star_story"
    assert "general_interview" in plan.candidate_template_ids
    assert not plan.fallback_used
    assert "star_story_builder" in plan.tools_to_consider
    assert "matched keywords" in plan.selection_reason
    assert plan.candidate_details[0].template_id == "star_story"
    assert plan.candidate_details[0].selected is True
    assert "star" in [keyword.lower() for keyword in plan.candidate_details[0].matched_keywords]
    assert any(item.rejected_reason for item in plan.candidate_details if item.template_id != "star_story")
    assert plan.route_trace[0].stage == "rule_recall"
    assert plan.route_trace[-1].stage == "rule_select"


def test_router_planner_uses_llm_router_when_available_and_confident():
    model = FakeStructuredRouterModel(
        {
            "template_id": "answer_review",
            "reason": "The user is explicitly asking to improve and review an interview answer.",
            "confidence": 0.91,
        }
    )
    planner = RouterPlanner.default(chat_model=model)

    plan = planner.create_plan(
        "请帮我优化这段项目回答，并给我具体 feedback",
        history=[MessageRecord(role="user", content="这是我的初稿")],
    )

    assert plan.template_id == "answer_review"
    assert plan.selection_strategy == "rule_plus_llm_rerank"
    assert "answer_review" in plan.candidate_template_ids
    assert plan.selection_confidence == 0.91
    assert not plan.fallback_used
    assert model.invocations
    assert any(item.selected for item in plan.candidate_details if item.template_id == "answer_review")
    assert any("LLM rerank" in (item.rejected_reason or "") for item in plan.candidate_details if item.template_id != "answer_review")
    assert plan.route_trace[-1].stage == "llm_rerank"


def test_router_planner_falls_back_to_default_template_when_no_router_matches():
    model = FakeStructuredRouterModel(
        {
            "template_id": "unknown_template",
            "reason": "Invalid output",
            "confidence": 0.99,
        }
    )
    planner = RouterPlanner.default(chat_model=model)

    plan = planner.create_plan("你好", history=[])

    assert plan.template_id == "general_interview"
    assert plan.selection_strategy == "fallback_default"
    assert plan.candidate_template_ids == []
    assert plan.fallback_used is True
    assert plan.selection_confidence == 0.0
    assert plan.candidate_details == []
    assert plan.route_trace[-1].stage == "default_fallback"


def test_router_planner_limits_candidates_to_top_k():
    planner = RouterPlanner.default(top_k_candidates=2)

    plan = planner.create_plan("请帮我准备一段项目经历回答并做 feedback 优化", history=[])

    assert len(plan.candidate_template_ids) <= 2
    assert plan.template_id in plan.candidate_template_ids
    assert len(plan.candidate_details) <= 2


def test_router_planner_uses_rule_fallback_when_llm_rerank_is_invalid_but_candidates_exist():
    model = FakeStructuredRouterModel(
        {
            "template_id": "not_in_candidates",
            "reason": "Bad rerank output",
            "confidence": 0.98,
        }
    )
    planner = RouterPlanner.default(chat_model=model)

    plan = planner.create_plan("请帮我优化这段回答并给 feedback", history=[])

    assert plan.template_id == "answer_review"
    assert plan.selection_strategy == "rule_based_fallback"
    assert plan.fallback_used is True
    assert any(item.stage == "llm_rerank_failed" for item in plan.route_trace)
    assert any(item.rejected_reason for item in plan.candidate_details if item.template_id != plan.template_id)


