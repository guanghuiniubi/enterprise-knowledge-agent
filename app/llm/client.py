from openai import OpenAI
from langsmith import traceable
from app.core.config import settings
from app.core.governance import governance_manager
from app.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.llm_model

    @traceable(name="llm_chat", run_type="llm")
    def chat(self, system_prompt: str, user_prompt: str) -> str:
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
        return resp.choices[0].message.content or ""

    @traceable(name="llm_chat_messages", run_type="llm")
    def chat_messages(self, messages: list[dict[str, str]]) -> str:
        resp = governance_manager.execute_llm(
            "chat_messages",
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            ),
        )
        return resp.choices[0].message.content or ""

    @traceable(name="llm_chat_with_tools", run_type="llm")
    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
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


llm_client = OpenAICompatibleLLM()
