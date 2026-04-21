from app.rag.vector_retriever import vector_retriever
from app.router.intent_router import IntentRouter
from app.schemas.agent import AgentResult
from app.tools.org_tool import org_tool
from app.tools.ticket_tool import ticket_tool
from app.tools.workflow_tool import workflow_tool
from app.llm.client import llm_client
from app.agent.clarifier import clarifier
from app.observability.tracing import traceable


class AgentOrchestrator:
    def __init__(self):
        self.router = IntentRouter()
        self.retriever = vector_retriever

    @traceable(name="agent_orchestrator_run")
    def run(self, question: str, context: str = "") -> AgentResult:
        route_result = self.router.route(question=question, context=context)

        if route_result.route == "clarification":
            clarification_question = clarifier.generate(route_result.missing_slots)
            return AgentResult(
                answer=clarification_question,
                route="clarification",
                need_clarification=True,
                clarification_question=clarification_question,
                debug={
                    "route_reason": route_result.reason,
                    "missing_slots": route_result.missing_slots
                }
            )

        if route_result.route == "ticket_query":
            tool_output = ticket_tool.query(question)
            return AgentResult(
                answer=tool_output["message"],
                route=route_result.route,
                tool_calls=[{
                    "tool_name": "ticket_tool",
                    "tool_input": {"question": question},
                    "tool_output": tool_output
                }],
                debug={"route_reason": route_result.reason}
            )

        if route_result.route == "org_query":
            tool_output = org_tool.query(question)
            return AgentResult(
                answer=tool_output["message"],
                route=route_result.route,
                tool_calls=[{
                    "tool_name": "org_tool",
                    "tool_input": {"question": question},
                    "tool_output": tool_output
                }],
                debug={"route_reason": route_result.reason}
            )

        if route_result.route == "workflow_query":
            tool_output = workflow_tool.query(question)
            return AgentResult(
                answer=tool_output["message"],
                route=route_result.route,
                tool_calls=[{
                    "tool_name": "workflow_tool",
                    "tool_input": {"question": question},
                    "tool_output": tool_output
                }],
                debug={"route_reason": route_result.reason}
            )

        docs = self.retriever.search(question, top_k=2)
        if not docs:
            return AgentResult(
                answer="我暂时没有在知识库中找到相关内容，建议你换个问法或补充更多背景。",
                route="knowledge_qa",
                debug={"route_reason": route_result.reason}
            )

        citations = []
        context_blocks = []
        for idx, doc in enumerate(docs, start=1):
            snippet = doc["content"][:200]
            citations.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "snippet": snippet
            })
            context_blocks.append(f"[文档{idx}] 标题: {doc['title']}\n内容: {doc['content']}")

        system_prompt = (
            "你是企业内部知识问答助手。"
            "请严格基于提供的知识库内容回答，不要编造。"
            "如果知识不足，请明确说明。"
            "回答要简洁、准确，并尽量引用依据。"
        )

        user_prompt = (
            f"会话上下文：\n{context or '无'}\n\n"
            f"用户问题：{question}\n\n"
            f"知识库片段：\n{chr(10).join(context_blocks)}\n\n"
            "请基于以上内容回答。"
        )

        answer = llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)

        return AgentResult(
            answer=answer,
            route="knowledge_qa",
            citations=citations,
            debug={"route_reason": route_result.reason}
        )


agent_orchestrator = AgentOrchestrator()
