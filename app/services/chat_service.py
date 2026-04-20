from app.agent.orchestrator import agent_orchestrator
from app.memory.session_store import session_store
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ToolCall


class ChatService:
    def chat(self, req: ChatRequest) -> ChatResponse:
        session_store.add_message(req.session_id, "user", req.question)

        result = agent_orchestrator.run(req.question)

        session_store.add_message(req.session_id, "assistant", result.answer)

        return ChatResponse(
            answer=result.answer,
            route=result.route,
            citations=[Citation(**item) for item in result.citations],
            tool_calls=[ToolCall(**item) for item in result.tool_calls],
            session_id=req.session_id,
            debug=result.debug
        )


chat_service = ChatService()
