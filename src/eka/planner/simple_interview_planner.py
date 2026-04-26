from __future__ import annotations

from eka.core.base import BasePlanner
from eka.core.types import MessageRecord, PlanRecord


class SimpleInterviewPlanner(BasePlanner):
    """A lightweight planner that turns user intent into an actionable plan summary.

    这个类代表项目最早期、最直接的 planner 思路：
    - 不做模板注册
    - 不做 recall / rerank
    - 不做 LLM route
    - 直接根据关键词拼一个 `PlanRecord`

    它非常适合做第一版原型，因为：
    - 实现简单
    - 可控性强
    - 调试成本低

    但它也有明显局限：
    - 业务模板一多就会膨胀
    - route 逻辑和 plan 构造耦合在一起
    - 不适合复杂工程扩展

    所以你现在项目里引入 `RouterPlanner`，本质上就是从这个“单体式 planner”
    向“工程化 planning pipeline”演进。
    """

    def create_plan(self, user_input: str, history: list[MessageRecord]) -> PlanRecord:
        """Generate a simple plan directly from keywords and recent history.

        这一步没有候选模板、也没有 rerank。
        它做的事情可以简单理解成：
        - 看看用户输入里有哪些关键词
        - 推测哪些工具可能有帮助
        - 拼一个 reasoning_summary 和 search_queries
        """
        text = user_input.lower()
        tools_to_consider: list[str] = []

        # 这些 if 就是最朴素的“意图识别”方式：
        # 某些关键词出现，就认为某类工具更有可能派上用场。
        if any(keyword in text for keyword in ["star", "项目", "经历", "behavior", "behaviour"]):
            tools_to_consider.append("star_story_builder")
        if any(keyword in text for keyword in ["准备", "计划", "清单", "roadmap", "checklist"]):
            tools_to_consider.append("interview_checklist")
        if any(keyword in text for keyword in ["回答", "review", "改进", "优化", "feedback"]):
            tools_to_consider.append("answer_rubric")

        # 没有明确命中时，给一个默认工具，避免 plan 太空。
        if not tools_to_consider:
            tools_to_consider.append("interview_checklist")

        # 为了让 plan 更像“结合上下文”的结果，这里只看最近两条历史消息。
        recent_context = "；".join(message.content for message in history[-2:]) if history else "无历史上下文"

        # 这里的 search query 也很简单：
        # 一个直接沿用用户输入，一个额外补上英文 AI Agent interview 检索词。
        search_queries = [user_input, f"AI Agent interview {user_input}"]
        reasoning_summary = (
            f"目标是帮助用户准备与“{user_input}”相关的面试表达，"
            f"先结合最近对话上下文（{recent_context}）判断意图，"
            f"再检索知识库并视情况调用工具：{', '.join(tools_to_consider)}。"
        )

        # 最终仍然输出统一的 `PlanRecord`，
        # 这也是为什么后续把 SimplePlanner 换成 RouterPlanner 时，executor 基本不用改。
        return PlanRecord(
            objective=f"为面试问题提供结构化辅导：{user_input}",
            reasoning_summary=reasoning_summary,
            search_queries=search_queries,
            tools_to_consider=tools_to_consider,
        )

