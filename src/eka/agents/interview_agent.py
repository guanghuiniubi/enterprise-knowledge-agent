from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from eka.config import build_chat_model, get_settings
from eka.core.base import BaseAgent, BaseExecutor, BaseMemory, BasePlanner, BaseRetriever, BaseSessionStore, BaseTool
from eka.core.types import ExecutionResult, TraceEvent
from eka.executor import LangGraphAgentExecutor
from eka.memory import SessionMemory
from eka.planner import RouterPlanner
from eka.retrievers import KeywordKnowledgeBaseRetriever
from eka.session import InMemorySessionStore
from eka.tools import build_default_tools


class InterviewAssistantAgent(BaseAgent):
    name = "interview_assistant"
    description = "A learning-oriented interview assistant agent powered by LangChain and LangGraph."

    def __init__(self, executor: BaseExecutor) -> None:
        self.executor = executor

    def respond(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        return self.executor.invoke(user_input=user_input, session_id=session_id)

    def stream(self, user_input: str, session_id: str = "default") -> Iterator[TraceEvent]:
        yield from self.executor.stream(user_input=user_input, session_id=session_id)

    @classmethod
    def create_default(
        cls,
        *,
        chat_model: Any | None = None,
        session_store: BaseSessionStore | None = None,
        memory: BaseMemory | None = None,
        planner: BasePlanner | None = None,
        retriever: BaseRetriever | None = None,
        tools: list[BaseTool] | None = None,
    ) -> "InterviewAssistantAgent":
        settings = get_settings()
        resolved_session_store = session_store or InMemorySessionStore()
        resolved_memory = memory or SessionMemory(resolved_session_store)
        resolved_chat_model = chat_model or build_chat_model(settings)
        resolved_planner = planner or RouterPlanner.default(chat_model=resolved_chat_model)
        resolved_retriever = retriever or KeywordKnowledgeBaseRetriever(settings.knowledge_base_dir)
        resolved_tools = tools or build_default_tools()
        executor = LangGraphAgentExecutor(
            chat_model=resolved_chat_model,
            memory=resolved_memory,
            planner=resolved_planner,
            retriever=resolved_retriever,
            tools=resolved_tools,
        )
        return cls(executor=executor)


def create_interview_agent(**kwargs: Any) -> InterviewAssistantAgent:
    return InterviewAssistantAgent.create_default(**kwargs)

