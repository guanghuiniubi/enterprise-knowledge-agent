import json
from app.llm.client import llm_client
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
        system_prompt = (
            "你是企业知识问答系统的意图分类器。"
            "请根据用户问题和上下文，将问题分类到以下类别之一：\n"
            "1. knowledge_qa: 企业制度、知识文档、FAQ类问题\n"
            "2. ticket_query: 工单状态、处理进度查询\n"
            "3. org_query: 组织架构、部门负责人、联系人查询\n"
            "4. workflow_query: 审批流程、流程节点、办理步骤查询\n"
            "5. clarification: 用户信息不足，必须先追问再继续\n\n"
            "如果问题缺少关键参数，例如用户问“我的工单怎么样了”但没有工单号，"
            "或者“审批到哪一步了”但没有业务单号，可以返回 clarification。\n\n"
            "输出必须是JSON，格式如下："
            '{"route":"knowledge_qa","reason":"...","missing_slots":["ticket_id"]}'
        )

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
