import json

from app.agent.orchestrator import agent_orchestrator
from app.core.governance import governance_manager
from app.memory.session_store import session_store
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ToolCall
from app.observability.tracing import traceable


class ChatService:
    def _request_key(self, req: ChatRequest) -> str:
        return req.user_id or req.session_id

    @traceable(name="chat_service_chat")
    def chat(self, req: ChatRequest) -> ChatResponse:
        governance_manager.enforce_request_rate_limit(self._request_key(req))
        context = session_store.format_recent_context(req.session_id, limit=6)
        session_store.add_message(req.session_id, "user", req.question, user_id=req.user_id)
        result = agent_orchestrator.run(req.question, context=context)

        session_store.add_message(req.session_id, "assistant", result.answer, user_id=req.user_id)

        return ChatResponse(
            answer=result.answer,
            route=result.route,
            citations=[Citation(**item) for item in result.citations],
            tool_calls=[ToolCall(**item) for item in result.tool_calls],
            agent_steps=result.agent_steps,
            session_id=req.session_id,
            need_clarification=result.need_clarification,
            clarification_question=result.clarification_question,
            debug=result.debug
        )

    @traceable(name="chat_service_chat_stream")
    def chat_stream(self, req: ChatRequest):
        governance_manager.enforce_request_rate_limit(self._request_key(req))
        context = session_store.format_recent_context(req.session_id, limit=6)
        session_store.add_message(req.session_id, "user", req.question, user_id=req.user_id)

        for event in agent_orchestrator.run_stream(req.question, context=context):
            payload = {**event}
            if event["type"] == "final":
                result = event["result"]
                session_store.add_message(req.session_id, "assistant", result.answer, user_id=req.user_id)
                payload = {
                    "type": "final",
                    "result": {
                        "answer": result.answer,
                        "route": result.route,
                        "citations": result.citations,
                        "tool_calls": result.tool_calls,
                        "agent_steps": result.agent_steps,
                        "session_id": req.session_id,
                        "need_clarification": result.need_clarification,
                        "clarification_question": result.clarification_question,
                        "debug": result.debug,
                    }
                }

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


chat_service = ChatService()
