import json
import time

from app.agent.orchestrator import agent_orchestrator
from app.core.governance import governance_manager
from app.core.request_context import set_request_context
from app.memory.session_store import session_store
from app.observability.metrics import observability_manager
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ToolCall
from app.observability.tracing import traceable
from app.security.access_control import AccessContext
from app.security.content_guard import content_guard


class ChatService:
    STREAM_CHUNK_SIZE = 24

    def _request_key(self, req: ChatRequest) -> str:
        return req.user_id or req.session_id

    def _access_context(self, req: ChatRequest) -> AccessContext:
        return AccessContext.from_payload(
            user_id=req.user_id,
            roles=req.user_roles,
            departments=req.user_departments,
            clearance_level=req.clearance_level,
        )

    def _safe_guard_response(self, req: ChatRequest, reasons: list[str]) -> ChatResponse:
        answer = "抱歉，我不能协助绕过系统提示词、工具协议或安全策略。你可以直接提出面试知识相关问题。"
        return ChatResponse(
            answer=answer,
            route="security_refusal",
            citations=[],
            tool_calls=[],
            agent_steps=[{"step": 1, "thought": "命中 prompt injection 防护", "action": "block_input", "reasons": reasons}],
            session_id=req.session_id,
            debug={"security": {"blocked": True, "reasons": reasons}},
        )

    def _apply_output_filter(self, answer: str) -> tuple[str, list[str]]:
        decision = content_guard.filter_output(answer)
        return decision.sanitized_text, decision.reasons

    def _tool_failure_count(self, tool_calls: list[dict]) -> int:
        failures = 0
        for item in tool_calls:
            output = item.get("tool_output", {}) if isinstance(item, dict) else {}
            if isinstance(output, dict) and output.get("error"):
                failures += 1
        return failures

    def _iter_answer_deltas(self, answer: str):
        if not answer:
            return

        buffer = ""
        split_tokens = {"\n", "。", "！", "？", "；", "：", ".", "!", "?", ";", ":", " ", "，", ",", "、"}
        for char in answer:
            buffer += char
            if len(buffer) >= self.STREAM_CHUNK_SIZE and char in split_tokens:
                yield buffer
                buffer = ""

        if buffer:
            yield buffer

    def _sse_payload(self, payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @traceable(name="chat_service_chat")
    def chat(self, req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        set_request_context(session_id=req.session_id, user_id=req.user_id)

        governance_manager.enforce_request_rate_limit(self._request_key(req))
        input_decision = content_guard.inspect_user_input(req.question)
        observability_manager.record_prompt_injection_check(
            hit=bool(input_decision.reasons),
            blocked=input_decision.blocked,
        )
        if input_decision.blocked:
            observability_manager.record_security_block()
            observability_manager.record_chat_request(
                route="security_refusal",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                fallback=False,
                step_count=1,
                tool_calls=0,
                tool_failures=0,
            )
            return self._safe_guard_response(req, input_decision.reasons)

        normalized_question = input_decision.sanitized_text or req.question
        context = session_store.format_recent_context(req.session_id, limit=6)
        session_store.add_message(req.session_id, "user", normalized_question, user_id=req.user_id)
        result = agent_orchestrator.run(
            normalized_question,
            context=context,
            access_context=self._access_context(req),
        )

        filtered_answer, output_reasons = self._apply_output_filter(result.answer)
        result.answer = filtered_answer
        if output_reasons:
            observability_manager.record_output_filter_hit()
        result.debug = {
            **(result.debug or {}),
            "security": {
                "blocked_input": False,
                "input_reasons": input_decision.reasons,
                "output_reasons": output_reasons,
            },
        }

        session_store.add_message(req.session_id, "assistant", result.answer, user_id=req.user_id)
        observability_manager.record_chat_request(
            route=result.route,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            fallback=bool((result.debug or {}).get("fallback")),
            step_count=len(result.agent_steps),
            tool_calls=len(result.tool_calls),
            tool_failures=self._tool_failure_count(result.tool_calls),
        )

        return ChatResponse(
            answer=result.answer,
            route=result.route,
            citations=[Citation(**item) for item in result.citations],
            tool_calls=[ToolCall(**item) for item in result.tool_calls],
            agent_steps=result.agent_steps,
            session_id=req.session_id,
            need_clarification=result.need_clarification,
            clarification_question=result.clarification_question,
            debug=result.debug,
        )

    @traceable(name="chat_service_chat_stream")
    def chat_stream(self, req: ChatRequest):
        started = time.perf_counter()
        set_request_context(session_id=req.session_id, user_id=req.user_id)

        governance_manager.enforce_request_rate_limit(self._request_key(req))
        input_decision = content_guard.inspect_user_input(req.question)
        observability_manager.record_prompt_injection_check(
            hit=bool(input_decision.reasons),
            blocked=input_decision.blocked,
        )
        if input_decision.blocked:
            observability_manager.record_security_block()
            observability_manager.record_stream_request(
                route="security_refusal",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                fallback=False,
                step_count=1,
                tool_calls=0,
                tool_failures=0,
            )
            payload = self._safe_guard_response(req, input_decision.reasons)
            for delta in self._iter_answer_deltas(payload.answer):
                yield self._sse_payload({"type": "answer_delta", "delta": delta})
            yield self._sse_payload({"type": "final", "result": payload.model_dump()})
            return

        normalized_question = input_decision.sanitized_text or req.question
        context = session_store.format_recent_context(req.session_id, limit=6)
        session_store.add_message(req.session_id, "user", normalized_question, user_id=req.user_id)
        streamed_answer_parts: list[str] = []

        for event in agent_orchestrator.run_stream(
                normalized_question,
                context=context,
                access_context=self._access_context(req),
        ):
            payload = {**event}
            if event["type"] == "answer_delta":
                delta = event.get("delta", "") or ""
                streamed_answer_parts.append(delta)
                yield self._sse_payload({"type": "answer_delta", "delta": delta})
                continue

            if event["type"] == "final":
                result = event["result"]
                raw_answer = result.answer or "".join(streamed_answer_parts)
                filtered_answer, output_reasons = self._apply_output_filter(raw_answer)
                result.answer = filtered_answer
                if output_reasons:
                    observability_manager.record_output_filter_hit()
                result.debug = {
                    **(result.debug or {}),
                    "security": {
                        "blocked_input": False,
                        "input_reasons": input_decision.reasons,
                        "output_reasons": output_reasons,
                    },
                }
                session_store.add_message(req.session_id, "assistant", result.answer, user_id=req.user_id)
                observability_manager.record_stream_request(
                    route=result.route,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    fallback=bool((result.debug or {}).get("fallback")),
                    step_count=len(result.agent_steps),
                    tool_calls=len(result.tool_calls),
                    tool_failures=self._tool_failure_count(result.tool_calls),
                )
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
                    },
                }

                streamed_answer = "".join(streamed_answer_parts)
                if streamed_answer_parts:
                    if streamed_answer != result.answer:
                        yield self._sse_payload({"type": "answer_replace", "answer": result.answer})
                else:
                    for delta in self._iter_answer_deltas(result.answer):
                        yield self._sse_payload({"type": "answer_delta", "delta": delta})

            yield self._sse_payload(payload)


chat_service = ChatService()
