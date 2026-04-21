from app.schemas.agent import RouteResult
from app.router.llm_router import llm_router
from app.observability.tracing import traceable


class IntentRouter:
    @traceable(name="intent_router")
    def route(self, question: str, context: str = "") -> RouteResult:
        q = question.lower()

        if "工单号" in question:
            return RouteResult(route="ticket_query", reason="规则命中工单号")

        if "部门负责人" in question or "leader" in q:
            return RouteResult(route="org_query", reason="规则命中部门负责人")

        if "审批流" in question:
            return RouteResult(route="workflow_query", reason="规则命中审批流")

        return llm_router.route(question=question, context=context)
