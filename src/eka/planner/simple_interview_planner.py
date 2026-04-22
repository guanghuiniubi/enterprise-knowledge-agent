from __future__ import annotations

from eka.core.base import BasePlanner
from eka.core.types import MessageRecord, PlanRecord


class SimpleInterviewPlanner(BasePlanner):
    """A lightweight planner that turns user intent into an actionable plan summary."""

    def create_plan(self, user_input: str, history: list[MessageRecord]) -> PlanRecord:
        text = user_input.lower()
        tools_to_consider: list[str] = []

        if any(keyword in text for keyword in ["star", "项目", "经历", "behavior", "behaviour"]):
            tools_to_consider.append("star_story_builder")
        if any(keyword in text for keyword in ["准备", "计划", "清单", "roadmap", "checklist"]):
            tools_to_consider.append("interview_checklist")
        if any(keyword in text for keyword in ["回答", "review", "改进", "优化", "feedback"]):
            tools_to_consider.append("answer_rubric")

        if not tools_to_consider:
            tools_to_consider.append("interview_checklist")

        recent_context = "；".join(message.content for message in history[-2:]) if history else "无历史上下文"
        search_queries = [user_input, f"AI Agent interview {user_input}"]
        reasoning_summary = (
            f"目标是帮助用户准备与“{user_input}”相关的面试表达，"
            f"先结合最近对话上下文（{recent_context}）判断意图，"
            f"再检索知识库并视情况调用工具：{', '.join(tools_to_consider)}。"
        )
        return PlanRecord(
            objective=f"为面试问题提供结构化辅导：{user_input}",
            reasoning_summary=reasoning_summary,
            search_queries=search_queries,
            tools_to_consider=tools_to_consider,
        )

