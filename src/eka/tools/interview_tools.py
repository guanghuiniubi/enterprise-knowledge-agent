from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eka.core.base import BaseTool


class InterviewChecklistInput(BaseModel):
    topic: str = Field(description="要准备的面试主题")
    seniority: str = Field(default="mid", description="候选人级别，例如 junior/mid/senior")


class InterviewChecklistTool(BaseTool):
    name = "interview_checklist"
    description = "为指定主题生成面试准备清单。"
    args_schema = InterviewChecklistInput

    def run(self, **kwargs: Any) -> str:
        payload = InterviewChecklistInput(**kwargs)
        checklist = [
            f"1. 用 1 分钟说明你对 {payload.topic} 的核心理解。",
            f"2. 准备 2 个与 {payload.topic} 相关的真实项目案例。",
            "3. 明确关键技术取舍、性能瓶颈和排障方法。",
            "4. 练习追问：为什么这样设计？替代方案是什么？",
            f"5. 按 {payload.seniority} 级别补充系统设计深度与业务影响。",
        ]
        return "\n".join(checklist)


class StarStoryInput(BaseModel):
    experience: str = Field(description="要转换为 STAR 表达的经历")
    target_role: str = Field(default="AI Agent 工程师", description="目标岗位")


class StarStoryBuilderTool(BaseTool):
    name = "star_story_builder"
    description = "将项目经历整理为 STAR 面试表达模板。"
    args_schema = StarStoryInput

    def run(self, **kwargs: Any) -> str:
        payload = StarStoryInput(**kwargs)
        return (
            f"S（情境）: 在申请 {payload.target_role} 过程中，你可以先说明背景：{payload.experience}\n"
            "T（任务）: 明确你的目标、指标或待解决问题。\n"
            "A（行动）: 说明你如何设计 Agent 架构、如何验证效果、如何处理异常。\n"
            "R（结果）: 用量化结果或学习收获收尾，例如效率提升、命中率提升、迭代速度提升。"
        )


class AnswerRubricInput(BaseModel):
    question: str = Field(description="面试问题")
    answer: str = Field(description="候选人的回答")


class AnswerRubricTool(BaseTool):
    name = "answer_rubric"
    description = "从结构、深度、量化和反思四个维度评估回答质量。"
    args_schema = AnswerRubricInput

    def run(self, **kwargs: Any) -> str:
        payload = AnswerRubricInput(**kwargs)
        answer = payload.answer.strip()
        structure = "良好" if any(marker in answer for marker in ["首先", "其次", "最后", "1."]) else "一般"
        quant = "良好" if any(token in answer for token in ["%", "倍", "ms", "QPS", "指标"]) else "可加强"
        depth = "良好" if any(token in answer for token in ["权衡", "trade-off", "异常", "回退", "监控"]) else "可加强"
        return (
            f"问题：{payload.question}\n"
            f"结构化表达：{structure}\n"
            f"技术深度：{depth}\n"
            f"量化结果：{quant}\n"
            "建议：补充方案取舍、失败案例和业务结果，会更像高级 Agent 工程师的回答。"
        )


def build_default_tools() -> list[BaseTool]:
    return [InterviewChecklistTool(), StarStoryBuilderTool(), AnswerRubricTool()]

