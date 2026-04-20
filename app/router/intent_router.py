from app.schemas.agent import RouteResult


class IntentRouter:
    def route(self, question: str) -> RouteResult:
        q = question.lower()

        if "工单" in question or "ticket" in q:
            return RouteResult(route="ticket_query", reason="命中工单关键词")

        if "组织" in question or "部门" in question or "leader" in q:
            return RouteResult(route="org_query", reason="命中组织信息关键词")

        if "审批" in question or "流程" in question:
            return RouteResult(route="workflow_query", reason="命中流程关键词")

        return RouteResult(route="knowledge_qa", reason="默认走知识问答")
