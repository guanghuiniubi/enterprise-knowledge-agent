from app.rag.retriever import KnowledgeRetriever
from app.router.intent_router import IntentRouter
from app.schemas.agent import AgentResult
from app.tools.org_tool import org_tool
from app.tools.ticket_tool import ticket_tool
from app.tools.workflow_tool import workflow_tool


class AgentOrchestrator:
    def __init__(self):
        self.router = IntentRouter()
        self.retriever = KnowledgeRetriever()

    def run(self, question: str) -> AgentResult:
        route_result = self.router.route(question)

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
                route=route_result.route,
                debug={"route_reason": route_result.reason}
            )

        citations = []
        snippets = []
        for doc in docs:
            snippet = doc["content"][:80]
            snippets.append(f"《{doc['title']}》提到：{snippet}")
            citations.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "snippet": snippet
            })

        answer = "根据知识库内容，" + "；".join(snippets)

        return AgentResult(
            answer=answer,
            route=route_result.route,
            citations=citations,
            debug={"route_reason": route_result.reason}
        )


agent_orchestrator = AgentOrchestrator()
