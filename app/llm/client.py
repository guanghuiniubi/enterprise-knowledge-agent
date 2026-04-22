import time
from collections.abc import Iterator

from openai import OpenAI
from langsmith import traceable
from app.core.config import settings
from app.core.governance import governance_manager
from app.llm.base import BaseLLM
from app.observability.metrics import observability_manager


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.llm_model

    @traceable(name="llm_chat", run_type="llm")
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        started = time.perf_counter()
        try:
            resp = governance_manager.execute_llm(
                "chat",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                ),
            )
            observability_manager.record_llm_call(
                operation="chat",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=True,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            observability_manager.record_llm_call(
                operation="chat",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=False,
                error_kind=type(exc).__name__,
            )
            raise

    @traceable(name="llm_chat_messages", run_type="llm")
    def chat_messages(self, messages: list[dict[str, str]]) -> str:
        started = time.perf_counter()
        try:
            resp = governance_manager.execute_llm(
                "chat_messages",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                ),
            )
            observability_manager.record_llm_call(
                operation="chat_messages",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=True,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            observability_manager.record_llm_call(
                operation="chat_messages",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=False,
                error_kind=type(exc).__name__,
            )
            raise

    @traceable(name="llm_chat_messages_stream", run_type="llm")
    def chat_messages_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        started = time.perf_counter()
        response_stream = None
        try:
            response_stream = governance_manager.execute_llm(
                "chat_messages_stream",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                ),
            )
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
            observability_manager.record_llm_call(
                operation="chat_messages_stream",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            observability_manager.record_llm_call(
                operation="chat_messages_stream",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=False,
                error_kind=type(exc).__name__,
            )
            raise
        finally:
            if response_stream is not None and hasattr(response_stream, "close"):
                response_stream.close()

    @traceable(name="llm_chat_with_tools", run_type="llm")
    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        started = time.perf_counter()
        try:
            resp = governance_manager.execute_llm(
                "chat_with_tools",
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                ),
            )
            observability_manager.record_llm_call(
                operation="chat_with_tools",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=True,
            )
            message = resp.choices[0].message
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    })

            return {
                "content": message.content or "",
                "tool_calls": tool_calls,
            }
        except Exception as exc:  # noqa: BLE001
            observability_manager.record_llm_call(
                operation="chat_with_tools",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                success=False,
                error_kind=type(exc).__name__,
            )
            raise


llm_client = OpenAICompatibleLLM()
