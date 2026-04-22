import json
from app.llm.client import llm_client
from app.prompts.registry import prompt_registry
from app.schemas.agent import RouteResult
from app.observability.tracing import traceable


class LLMRouter:
    ALLOWED_ROUTES = {
        "knowledge_qa",
        "ticket_query",
        "org_query",
        "workflow_query",
        "clarification",
    }

    @traceable(name="llm_route_classification")
    def route(self, question: str, context: str = "") -> RouteResult:
        system_prompt = prompt_registry.render("route_classifier_system")

        user_prompt = f"上下文：\n{context or '无'}\n\n用户问题：{question}"

        raw = llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)

        try:
            data = json.loads(raw)
            route = data.get("route", "knowledge_qa")
            if route not in self.ALLOWED_ROUTES:
                route = "knowledge_qa"
            return RouteResult(
                route=route,
                reason=data.get("reason", "LLM路由"),
                missing_slots=data.get("missing_slots", [])
            )
        except Exception:
            return RouteResult(
                route="knowledge_qa",
                reason=f"LLM路由解析失败，回退默认路由。raw={raw}",
                missing_slots=[]
            )


llm_router = LLMRouter()
