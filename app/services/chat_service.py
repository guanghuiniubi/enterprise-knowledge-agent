from app.agent.orchestrator import agent_orchestrator
from app.agent.slot_extractor import slot_extractor
from app.memory.session_store import session_store
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ToolCall
from app.observability.tracing import traceable
from app.services.session_state_service import session_state_service
from app.tools.ticket_tool import ticket_tool
from app.tools.workflow_tool import workflow_tool


class ChatService:
    @traceable(name="chat_service_chat")
    def chat(self, req: ChatRequest) -> ChatResponse:
        context = session_store.format_recent_context(req.session_id, limit=6)
        state = session_state_service.get(req.session_id)

        session_store.add_message(req.session_id, "user", req.question)

        # 1. 如果当前处于等待补槽状态，优先尝试从本轮输入中抽槽
        if state and state.status == "waiting_clarification" and state.pending_slots:
            extracted = slot_extractor.extract(req.question)
            merged_slots = dict(state.collected_slots or {})
            merged_slots.update(extracted)

            remaining_slots = [slot for slot in state.pending_slots if slot not in merged_slots]

            # 槽位补齐，继续执行原任务
            if not remaining_slots:
                if state.current_intent == "ticket_query":
                    tool_output = ticket_tool.query(
                        question=req.question,
                        ticket_id=merged_slots.get("ticket_id")
                    )
                    answer = tool_output["message"]

                    session_store.add_message(req.session_id, "assistant", answer)
                    session_state_service.save_completed_state(
                        session_id=req.session_id,
                        user_id=req.user_id,
                        current_intent="ticket_query",
                        collected_slots=merged_slots
                    )

                    return ChatResponse(
                        answer=answer,
                        route="ticket_query",
                        citations=[],
                        tool_calls=[ToolCall(
                            tool_name="ticket_tool",
                            tool_input={"ticket_id": merged_slots.get("ticket_id")},
                            tool_output=tool_output
                        )],
                        session_id=req.session_id,
                        need_clarification=False,
                        clarification_question=None,
                        debug={
                            "from_waiting_state": True,
                            "collected_slots": merged_slots
                        }
                    )

                if state.current_intent == "workflow_query":
                    tool_output = workflow_tool.query(
                        question=req.question,
                        workflow_id=merged_slots.get("workflow_id")
                    )
                    answer = tool_output["message"]

                    session_store.add_message(req.session_id, "assistant", answer)
                    session_state_service.save_completed_state(
                        session_id=req.session_id,
                        user_id=req.user_id,
                        current_intent="workflow_query",
                        collected_slots=merged_slots
                    )

                    return ChatResponse(
                        answer=answer,
                        route="workflow_query",
                        citations=[],
                        tool_calls=[ToolCall(
                            tool_name="workflow_tool",
                            tool_input={"workflow_id": merged_slots.get("workflow_id")},
                            tool_output=tool_output
                        )],
                        session_id=req.session_id,
                        need_clarification=False,
                        clarification_question=None,
                        debug={
                            "from_waiting_state": True,
                            "collected_slots": merged_slots
                        }
                    )

            # 槽位仍然不够，继续追问
            session_state_service.save_waiting_state(
                session_id=req.session_id,
                user_id=req.user_id,
                current_intent=state.current_intent,
                pending_slots=remaining_slots,
                collected_slots=merged_slots
            )

            answer = "我还需要更多信息才能继续，请继续补充。"
            session_store.add_message(req.session_id, "assistant", answer)

            return ChatResponse(
                answer=answer,
                route="clarification",
                session_id=req.session_id,
                need_clarification=True,
                clarification_question=answer,
                debug={
                    "from_waiting_state": True,
                    "remaining_slots": remaining_slots,
                    "collected_slots": merged_slots
                }
            )

        # 2. 正常进入 Agent 编排
        result = agent_orchestrator.run(req.question, context=context)

        # 3. 如果需要澄清，则落库 session state
        if result.need_clarification:
            current_intent = "knowledge_qa"
            missing_slots = result.debug.get("missing_slots", []) if result.debug else []

            # 尝试根据缺失槽位反推 intent
            if "ticket_id" in missing_slots:
                current_intent = "ticket_query"
            elif "workflow_id" in missing_slots:
                current_intent = "workflow_query"
            elif "department_name" in missing_slots:
                current_intent = "org_query"

            session_state_service.save_waiting_state(
                session_id=req.session_id,
                user_id=req.user_id,
                current_intent=current_intent,
                pending_slots=missing_slots,
                collected_slots={}
            )

        session_store.add_message(req.session_id, "assistant", result.answer)

        return ChatResponse(
            answer=result.answer,
            route=result.route,
            citations=[Citation(**item) for item in result.citations],
            tool_calls=[ToolCall(**item) for item in result.tool_calls],
            session_id=req.session_id,
            need_clarification=result.need_clarification,
            clarification_question=result.clarification_question,
            debug=result.debug
        )


chat_service = ChatService()
