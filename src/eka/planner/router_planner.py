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
    """A recalled candidate before final planner selection.

    你可以把它理解成“候选模板卡片”：
    - `template_id`：候选模板是谁
    - `score`：规则召回阶段给它打了多少分
    - `priority`：模板自己的静态优先级
    - `matched_keywords`：它是因为什么关键词被召回的

    注意它还不是最终选择结果，只是进入 rerank / fallback 阶段的候选集。
    """
    template_id: str
    score: int
    priority: int
    matched_keywords: list[str]


@dataclass(slots=True)
class RouteDecision:
    """Final routing decision passed into the selected plan template.

    一旦 RouterPlanner 决定了“最终采用哪个模板”，就会把决定结果放进这个对象里。
    这个对象除了记录最终模板，还会把整个路由过程里最关键的信息带下去：
    - 候选模板有哪些
    - 是通过什么策略选中的（rule / rerank / fallback）
    - 原因是什么
    - 是否发生了回退
    - 候选详情和 route trace 是什么

    这样 template 在生成 `PlanRecord` 时，就能把这些元数据一起写进去，
    后续 CLI / Trace / API / Eval 都能复用。
    """
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
    """Wrapper for LLM rerank results.

    为什么不直接返回 `RouteDecision | None`？
    因为真实工程里“没选出来”也很重要：
    - 是模型不支持 structured output？
    - 是调用失败？
    - 是置信度太低？
    - 还是模型选了非法模板？

    `failure_reason` 会被写入 route trace，帮助你调试 rerank 的失败原因。
    """
    decision: RouteDecision | None
    failure_reason: str | None = None


class BasePlanTemplate(ABC):
    """Base abstraction for reusable plan templates.

    这里的 template 不是“具体执行图”，而是“某一类规划任务的骨架”。
    例如：
    - `general_interview`
    - `star_story`
    - `answer_review`
    - `preparation_checklist`

    每个 template 负责：
    - 定义自己的语义定位（description / route_keywords / priority）
    - 被选中后把请求构造成统一的 `PlanRecord`

    这样可以把“选哪个模板”和“模板内部如何构造 plan”拆开。
    """

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
        """Return all registered templates.

        recall 阶段会对整个 registry 做扫描，因此这里返回的是“全集”。
        """
        return list(self._templates)

    def get(self, template_id: str) -> BasePlanTemplate:
        """Get a template by id, raising a clear error if it is missing."""
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
        """Turn the final route decision into a unified `PlanRecord`.

        这里是“模板执行阶段”，不是“路由阶段”。
        Router 已经决定了当前请求属于哪个模板；
        这个函数只负责把模板自己的风格和重点，组合成标准化的 `PlanRecord`。

        统一结构的好处是：
        - `LangGraphExecutor` 不需要关心模板细节
        - 后续替换 planner / template 时，对 executor 基本透明
        - trace / CLI / API 的消费方式也保持一致
        """
        recent_context = "；".join(message.content for message in history[-2:]) if history else "无历史上下文"
        # search queries 同样由模板来决定，这意味着不同模板可以有不同检索策略。
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
    """Deterministic router that recalls top-k candidate templates using keyword overlap.

    这里不要把它理解成“最终决策器”，它更像：
    - 一个低成本的召回器（recall）
    - 一个稳定的规则 fallback

    在真实工程里，规则路由通常不是为了完美理解语义，
    而是为了：
    - 快速缩小候选范围
    - 降低 LLM rerank 的成本
    - 在 LLM 失败时兜底
    """

    def recall(
        self,
        user_input: str,
        history: list[MessageRecord],
        templates: list[BasePlanTemplate],
        *,
        limit: int = 3,
    ) -> list[TemplateCandidate]:
        """Recall top-k candidate templates from all registered templates.

        这一步的目标不是“100% 选对最终模板”，而是“尽量把正确模板召回来”。

        逻辑非常直接：
        1. 对每个 template 看它的 `route_keywords` 在 user_input / recent history 中命中了多少
        2. 用命中数量作为 score
        3. 再按 template.priority 做二次排序
        4. 取 top-k

        这是典型的工程做法：
        先做轻量召回，再做重排序，而不是一上来就把所有模板交给大模型。
        """
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
        """Pick the top recalled candidate as the deterministic rule-based decision.

        当 LLM rerank 不存在或失败时，这个函数给出一个可落地的最终答案：
        直接选择 recall 阶段排名第一的候选。
        """
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
    """Semantic reranker over a pre-recalled candidate template set.

    这个类的职责非常明确：
    - 它不负责全量模板选择
    - 它只对 recall 后的候选集做“重排 / 语义判别”

    这比“让 LLM 直接在所有模板中选一个”更工程化，因为：
    - 成本更低
    - 控制更强
    - 更容易解释失败原因
    - 候选范围可被规则严格约束
    """

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
        """Rerank recalled candidates with an LLM and return either a decision or a failure reason.

        这里输出的是 `LLMRouteOutcome`，而不是直接扔异常。
        因为在真实系统中，rerank 失败是一种“正常情况”，不应该让整条链路崩掉。
        """
        if not candidates:
            return LLMRouteOutcome(decision=None, failure_reason="No candidates available for LLM rerank.")
        if not hasattr(self.model, "with_structured_output"):
            return LLMRouteOutcome(decision=None, failure_reason="Model does not support structured output.")

        # 这里要求模型必须输出结构化结果，避免脆弱的自然语言解析。
        structured_model = self.model.with_structured_output(LLMRouterSelection)
        recent_history = "\n".join(f"- {message.role}: {message.content}" for message in history[-3:]) or "- 无历史消息"
        # 候选集不仅给 template id，还把规则分数与命中关键词一起给到模型，
        # 这样模型是在“语义 + 规则上下文”的基础上 rerank，而不是盲选。
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

        # 即使模型返回了结构化结果，我们仍然要做程序侧校验：
        # - 选的模板是否在候选集中
        # - 置信度是否达标
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
    """Production-style planner with registry, top-k recall, LLM rerank, and fallback.

    这是当前项目里“真实工程版 planner”的核心入口。

    它把 planning 拆成了 4 个阶段：
    1. registry：有哪些模板可选
    2. recall：先召回 top-k 候选
    3. rerank：如果有 LLM，就对候选做语义重排
    4. fallback：如果 rerank 不可用，就退回规则结果；如果连候选都没有，就用默认模板

    这和真实检索 / 排序系统很像：
    recall -> rerank -> fallback
    """

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
        """Create a plan through recall, optional rerank, and fallback.

        这是整个 planner 最值得重点阅读的函数。

        你可以把它当成一条决策流水线：

        - recall：先找候选模板
        - llm_rerank：如果可用，让 LLM 在候选中选一个
        - rule_select：如果 LLM 失败，使用规则第一名
        - default_fallback：如果一个候选都没有，走默认模板

        同时，这个函数还会生成：
        - `candidate_details`
        - `route_trace`

        这样后续你不仅知道“选了谁”，还知道“为什么没选别人”。
        """
        candidates = self.rule_router.recall(
            user_input,
            history,
            self.registry.all(),
            limit=self.top_k_candidates,
        )

        # base_trace 记录 recall 阶段的结构化结果，是整个 route_trace 的起点。
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
            # rerank 成功时，最终决策来源于 LLM，
            # 但 candidate_details 仍然保留规则召回信息，用于解释“候选集长什么样”。
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
            # 如果 LLM 不可用、失败、低置信度或返回非法模板，
            # 则退回 rule recall 的 top-1 结果。
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
            # 连候选都没有召回出来时，走系统默认模板。
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
        """Build structured candidate records for CLI/trace/eval consumption.

        这里把 recall 阶段的候选集进一步标准化：
        - 谁被选中了
        - 谁没被选中
        - 没被选中的原因是什么

        这一步很关键，因为后续：
        - CLI 可以展示候选详情
        - Trace 可以记录 route 细节
        - Eval 可以比较 route 的稳定性
        """
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
