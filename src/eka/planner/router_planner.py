from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from eka.core.base import BasePlanner
from eka.core.types import MessageRecord, PlanCandidateRecord, PlanRecord, RouteTraceRecord


@dataclass(slots=True)
class TemplateCandidate:
    template_id: str
    score: int
    priority: int
    matched_keywords: list[str]


@dataclass(slots=True)
class RouteDecision:
    template_id: str
    selection_strategy: str
    selection_reason: str
    candidate_template_ids: list[str]
    selection_confidence: float | None = None
    fallback_used: bool = False
    candidate_details: list[PlanCandidateRecord] = field(default_factory=list)
    route_trace: list[RouteTraceRecord] = field(default_factory=list)


@dataclass(slots=True)
class LLMRouteOutcome:
    decision: RouteDecision | None
    failure_reason: str | None = None


class BasePlanTemplate(ABC):
    """Base abstraction for reusable plan templates."""

    template_id: str
    description: str
    route_keywords: tuple[str, ...]
    priority: int

    @abstractmethod
    def build_plan(self, user_input: str, history: list[MessageRecord], decision: RouteDecision) -> PlanRecord:
        raise NotImplementedError


class TemplateRegistry:
    """Registry for planner templates.

    真实工程里，模板不应该散落在多个 `if/elif` 中，而应该集中注册、统一管理，
    这样后续才能支持：
    - 模板扩展
    - 模板下线
    - 模板评估
    - 模板版本管理
    """

    def __init__(self, templates: list[BasePlanTemplate]) -> None:
        if not templates:
            raise ValueError("TemplateRegistry requires at least one plan template.")
        self._templates = templates
        self._templates_by_id = {template.template_id: template for template in templates}

    def all(self) -> list[BasePlanTemplate]:
        return list(self._templates)

    def get(self, template_id: str) -> BasePlanTemplate:
        try:
            return self._templates_by_id[template_id]
        except KeyError as exc:
            raise ValueError(f"Unknown plan template '{template_id}'.") from exc

    def ids(self) -> list[str]:
        return [template.template_id for template in self._templates]


@dataclass(slots=True)
class InterviewPlanTemplate(BasePlanTemplate):
    template_id: str
    description: str
    route_keywords: tuple[str, ...]
    tools_to_consider: tuple[str, ...]
    objective_prefix: str
    summary_focus: str
    priority: int = 0
    extra_search_queries: tuple[str, ...] = ()

    def build_plan(self, user_input: str, history: list[MessageRecord], decision: RouteDecision) -> PlanRecord:
        recent_context = "；".join(message.content for message in history[-2:]) if history else "无历史上下文"
        search_queries = [user_input, *self.extra_search_queries, f"AI Agent interview {user_input}"]
        reasoning_summary = (
            f"当前路由选择了模板“{self.template_id}”，"
            f"策略为 {decision.selection_strategy}，"
            f"候选模板：{', '.join(decision.candidate_template_ids) if decision.candidate_template_ids else '无'}。"
            f"原因：{decision.selection_reason or '未提供'}。"
            f"本轮重点是{self.summary_focus}；"
            f"最近对话上下文：{recent_context}。"
            f"建议优先结合这些工具：{', '.join(self.tools_to_consider) if self.tools_to_consider else '无'}。"
        )
        return PlanRecord(
            objective=f"{self.objective_prefix}：{user_input}",
            reasoning_summary=reasoning_summary,
            search_queries=search_queries,
            tools_to_consider=list(self.tools_to_consider),
            template_id=self.template_id,
            candidate_template_ids=decision.candidate_template_ids,
            candidate_details=decision.candidate_details,
            selection_strategy=decision.selection_strategy,
            selection_reason=decision.selection_reason,
            selection_confidence=decision.selection_confidence,
            fallback_used=decision.fallback_used,
            route_trace=decision.route_trace,
        )


class RuleBasedIntentRouter:
    """Deterministic router that recalls top-k candidate templates using keyword overlap."""

    def recall(
        self,
        user_input: str,
        history: list[MessageRecord],
        templates: list[BasePlanTemplate],
        *,
        limit: int = 3,
    ) -> list[TemplateCandidate]:
        text = self._normalize(user_input)
        history_text = self._normalize(" ".join(message.content for message in history[-3:]))
        candidates: list[TemplateCandidate] = []

        for template in templates:
            matched_keywords = [
                keyword
                for keyword in template.route_keywords
                if self._normalize(keyword) in text or self._normalize(keyword) in history_text
            ]
            score = len(matched_keywords)
            if score <= 0:
                continue
            candidates.append(
                TemplateCandidate(
                    template_id=template.template_id,
                    score=score,
                    priority=template.priority,
                    matched_keywords=matched_keywords,
                )
            )

        candidates.sort(key=lambda item: (item.score, item.priority), reverse=True)
        return candidates[:limit]

    def route(self, candidates: list[TemplateCandidate]) -> RouteDecision | None:
        if not candidates:
            return None
        top_candidate = candidates[0]
        confidence = min(0.55 + top_candidate.score * 0.15, 0.95)
        return RouteDecision(
            template_id=top_candidate.template_id,
            selection_strategy="rule_based",
            selection_reason=f"matched keywords: {', '.join(top_candidate.matched_keywords)}",
            candidate_template_ids=[candidate.template_id for candidate in candidates],
            selection_confidence=confidence,
            fallback_used=False,
        )

    def _normalize(self, text: str) -> str:
        return text.lower().strip()


class LLMRouterSelection(BaseModel):
    template_id: str = Field(description="The selected template id.")
    reason: str = Field(description="Why this template best fits the request.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")


class LLMIntentRouter:
    """Semantic reranker over a pre-recalled candidate template set."""

    def __init__(self, model: Any, confidence_threshold: float = 0.6) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold

    def route(
        self,
        user_input: str,
        history: list[MessageRecord],
        candidates: list[TemplateCandidate],
        registry: TemplateRegistry,
    ) -> LLMRouteOutcome:
        if not candidates:
            return LLMRouteOutcome(decision=None, failure_reason="No candidates available for LLM rerank.")
        if not hasattr(self.model, "with_structured_output"):
            return LLMRouteOutcome(decision=None, failure_reason="Model does not support structured output.")

        structured_model = self.model.with_structured_output(LLMRouterSelection)
        recent_history = "\n".join(f"- {message.role}: {message.content}" for message in history[-3:]) or "- 无历史消息"
        candidate_catalog = "\n".join(
            (
                f"- id={candidate.template_id}; "
                f"description={registry.get(candidate.template_id).description}; "
                f"rule_score={candidate.score}; "
                f"matched_keywords={', '.join(candidate.matched_keywords) or 'none'}"
            )
            for candidate in candidates
        )
        messages = [
            SystemMessage(
                content=(
                    "你是一个 Planner Router，需要对候选 plan templates 做 rerank，"
                    "从候选 template id 中选择最适合当前请求的一个，并返回结构化结果。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户输入：{user_input}\n"
                    f"最近历史：\n{recent_history}\n"
                    f"候选模板：\n{candidate_catalog}\n"
                    "请基于语义最匹配的任务规划方式，选择一个模板。"
                )
            ),
        ]

        try:
            selection = structured_model.invoke(messages)
        except Exception:
            return LLMRouteOutcome(decision=None, failure_reason="LLM rerank invocation failed.")

        candidate_ids = {candidate.template_id for candidate in candidates}
        if selection.template_id not in candidate_ids:
            return LLMRouteOutcome(
                decision=None,
                failure_reason=f"LLM selected template '{selection.template_id}' outside recalled candidates.",
            )
        if selection.confidence < self.confidence_threshold:
            return LLMRouteOutcome(
                decision=None,
                failure_reason=(
                    f"LLM confidence {selection.confidence:.2f} below threshold {self.confidence_threshold:.2f}."
                ),
            )
        return LLMRouteOutcome(
            decision=RouteDecision(
                template_id=selection.template_id,
                selection_strategy="rule_plus_llm_rerank",
                selection_reason=selection.reason,
                candidate_template_ids=[candidate.template_id for candidate in candidates],
                selection_confidence=selection.confidence,
                fallback_used=False,
            )
        )


class RouterPlanner(BasePlanner):
    """Production-style planner with registry, top-k recall, LLM rerank, and fallback."""

    def __init__(
        self,
        *,
        templates: list[BasePlanTemplate],
        fallback_template_id: str,
        top_k_candidates: int = 3,
        rule_router: RuleBasedIntentRouter | None = None,
        llm_router: LLMIntentRouter | None = None,
    ) -> None:
        self.registry = TemplateRegistry(templates)
        self.fallback_template_id = fallback_template_id
        self.top_k_candidates = top_k_candidates
        self.rule_router = rule_router or RuleBasedIntentRouter()
        self.llm_router = llm_router

        if fallback_template_id not in self.registry.ids():
            raise ValueError(f"Unknown fallback template '{fallback_template_id}'.")

    @classmethod
    def default(cls, *, chat_model: Any | None = None, top_k_candidates: int = 3) -> "RouterPlanner":
        templates = build_default_plan_templates()
        return cls(
            templates=templates,
            fallback_template_id="general_interview",
            top_k_candidates=top_k_candidates,
            rule_router=RuleBasedIntentRouter(),
            llm_router=LLMIntentRouter(chat_model) if chat_model is not None else None,
        )

    def create_plan(self, user_input: str, history: list[MessageRecord]) -> PlanRecord:
        candidates = self.rule_router.recall(
            user_input,
            history,
            self.registry.all(),
            limit=self.top_k_candidates,
        )
        base_trace = [
            RouteTraceRecord(
                stage="rule_recall",
                message=f"Recalled {len(candidates)} candidate templates.",
                data={
                    "candidates": [
                        {
                            "template_id": candidate.template_id,
                            "score": candidate.score,
                            "priority": candidate.priority,
                            "matched_keywords": candidate.matched_keywords,
                        }
                        for candidate in candidates
                    ]
                },
            )
        ]
        llm_outcome = self.llm_router.route(user_input, history, candidates, self.registry) if self.llm_router else None
        llm_decision = llm_outcome.decision if llm_outcome else None
        rule_decision = self.rule_router.route(candidates)

        if llm_decision is not None:
            final_decision = RouteDecision(
                template_id=llm_decision.template_id,
                selection_strategy=llm_decision.selection_strategy,
                selection_reason=llm_decision.selection_reason,
                candidate_template_ids=llm_decision.candidate_template_ids,
                selection_confidence=llm_decision.selection_confidence,
                fallback_used=False,
                candidate_details=self._build_candidate_details(
                    candidates,
                    selected_template_id=llm_decision.template_id,
                    non_selected_reason=f"Not selected by LLM rerank; '{llm_decision.template_id}' ranked higher.",
                ),
                route_trace=base_trace
                + [
                    RouteTraceRecord(
                        stage="llm_rerank",
                        message=f"LLM rerank selected template '{llm_decision.template_id}'.",
                        data={
                            "selected_template_id": llm_decision.template_id,
                            "confidence": llm_decision.selection_confidence,
                            "reason": llm_decision.selection_reason,
                        },
                    )
                ],
            )
            return self.registry.get(final_decision.template_id).build_plan(user_input, history, final_decision)

        if rule_decision is not None:
            fallback_from_llm = self.llm_router is not None
            decision = RouteDecision(
                template_id=rule_decision.template_id,
                selection_strategy="rule_based_fallback" if fallback_from_llm else rule_decision.selection_strategy,
                selection_reason=(
                    rule_decision.selection_reason
                    if not fallback_from_llm
                    else f"LLM rerank unavailable or below threshold; {rule_decision.selection_reason}"
                ),
                candidate_template_ids=rule_decision.candidate_template_ids,
                selection_confidence=rule_decision.selection_confidence,
                fallback_used=fallback_from_llm,
                candidate_details=self._build_candidate_details(
                    candidates,
                    selected_template_id=rule_decision.template_id,
                    non_selected_reason=f"Lower-ranked candidate than '{rule_decision.template_id}' in rule recall.",
                ),
                route_trace=base_trace
                + (
                    [
                        RouteTraceRecord(
                            stage="llm_rerank_failed",
                            message=llm_outcome.failure_reason or "LLM rerank did not produce a selection.",
                            data={"candidates": rule_decision.candidate_template_ids},
                        )
                    ]
                    if llm_outcome is not None
                    else []
                )
                + [
                    RouteTraceRecord(
                        stage="rule_select",
                        message=f"Rule router selected template '{rule_decision.template_id}'.",
                        data={
                            "selected_template_id": rule_decision.template_id,
                            "confidence": rule_decision.selection_confidence,
                            "reason": (
                                rule_decision.selection_reason
                                if not fallback_from_llm
                                else f"LLM rerank unavailable or below threshold; {rule_decision.selection_reason}"
                            ),
                        },
                    )
                ],
            )
            return self.registry.get(decision.template_id).build_plan(user_input, history, decision)

        fallback_decision = RouteDecision(
            template_id=self.fallback_template_id,
            selection_strategy="fallback_default",
            selection_reason="No candidate templates recalled; using default template.",
            candidate_template_ids=[],
            selection_confidence=0.0,
            fallback_used=True,
            candidate_details=[],
            route_trace=base_trace
            + (
                [
                    RouteTraceRecord(
                        stage="llm_rerank_failed",
                        message=llm_outcome.failure_reason or "LLM rerank skipped due to missing candidates.",
                    )
                ]
                if llm_outcome is not None
                else []
            )
            + [
                RouteTraceRecord(
                    stage="default_fallback",
                    message=f"Fallback to default template '{self.fallback_template_id}'.",
                    data={"reason": "No candidate templates recalled."},
                )
            ],
        )
        return self.registry.get(self.fallback_template_id).build_plan(user_input, history, fallback_decision)

    def _build_candidate_details(
        self,
        candidates: list[TemplateCandidate],
        *,
        selected_template_id: str,
        non_selected_reason: str,
    ) -> list[PlanCandidateRecord]:
        return [
            PlanCandidateRecord(
                template_id=candidate.template_id,
                score=candidate.score,
                priority=candidate.priority,
                matched_keywords=list(candidate.matched_keywords),
                selected=candidate.template_id == selected_template_id,
                rejected_reason=None if candidate.template_id == selected_template_id else non_selected_reason,
            )
            for candidate in candidates
        ]


def build_default_plan_templates() -> list[BasePlanTemplate]:
    return [
        InterviewPlanTemplate(
            template_id="general_interview",
            description="通用 AI Agent 面试规划模板，适合默认兜底和泛化问题。",
            route_keywords=("interview", "面试", "agent", "langgraph", "langchain", "设计", "架构", "准备"),
            tools_to_consider=("interview_checklist",),
            objective_prefix="为面试问题提供通用结构化辅导",
            summary_focus="先澄清问题，再给出结构化回答框架、关键知识点和准备建议",
            priority=10,
            extra_search_queries=("AI Agent interview system design",),
        ),
        InterviewPlanTemplate(
            template_id="star_story",
            description="将项目经历整理为 STAR 叙事，适合项目复盘、行为面试、经历表达。",
            route_keywords=("star", "项目", "经历", "behavior", "behaviour", "案例"),
            tools_to_consider=("star_story_builder", "interview_checklist"),
            objective_prefix="为项目经历提供 STAR 化面试辅导",
            summary_focus="突出情境、任务、行动、结果，并把经历说得更像面试表达",
            priority=30,
            extra_search_queries=("STAR interview storytelling",),
        ),
        InterviewPlanTemplate(
            template_id="answer_review",
            description="聚焦回答优化、点评与反馈，适合 review / feedback / 改进类问题。",
            route_keywords=("回答", "review", "改进", "优化", "feedback", "点评", "润色"),
            tools_to_consider=("answer_rubric",),
            objective_prefix="为候选人回答提供复盘与改进建议",
            summary_focus="评估当前回答质量，指出结构、深度、量化与反思上的改进点",
            priority=40,
            extra_search_queries=("interview answer rubric",),
        ),
        InterviewPlanTemplate(
            template_id="preparation_checklist",
            description="聚焦准备计划、清单与 roadmap，适合面试准备方案设计。",
            route_keywords=("准备", "计划", "清单", "roadmap", "checklist", "复习"),
            tools_to_consider=("interview_checklist",),
            objective_prefix="为用户制定面试准备计划",
            summary_focus="输出阶段性准备清单、知识补齐路线和练习建议",
            priority=20,
            extra_search_queries=("AI Agent interview preparation checklist",),
        ),
    ]
