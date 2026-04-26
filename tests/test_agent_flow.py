from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage

from eka.agents import InterviewAssistantAgent
from eka.executor import LangGraphAgentExecutor
from eka.memory import SessionMemory
from eka.planner import SimpleInterviewPlanner
from eka.retrievers import KeywordKnowledgeBaseRetriever
from eka.session import InMemorySessionStore
from eka.tools import InterviewChecklistTool


class FakeToolAwareChatModel:
	def __init__(self, responses: Sequence[AIMessage]):
		self._responses = list(responses)
		self.bound_tools = []
		self.invocations = []

	def bind_tools(self, tools):
		self.bound_tools = list(tools)
		return self

	def invoke(self, messages):
		self.invocations.append(messages)
		if not self._responses:
			raise AssertionError("No fake responses left for the chat model.")
		return self._responses.pop(0)


def build_agent(model, knowledge_dir=None):
	session_store = InMemorySessionStore()
	executor = LangGraphAgentExecutor(
		chat_model=model,
		memory=SessionMemory(session_store),
		planner=SimpleInterviewPlanner(),
		retriever=KeywordKnowledgeBaseRetriever(knowledge_dir),
		tools=[InterviewChecklistTool()],
	)
	return InterviewAssistantAgent(executor), session_store


def test_agent_completes_a_basic_turn_with_trace_and_memory(tmp_path):
	(tmp_path / "notes.md").write_text("LangGraph interview focus on planning and retrieval.", encoding="utf-8")
	model = FakeToolAwareChatModel([AIMessage(content="建议你从规划、记忆、工具调用三个方面回答。")])
	agent, session_store = build_agent(model, tmp_path)

	result = agent.respond("怎么介绍 LangGraph Agent 设计？", session_id="s1")

	assert "规划" in result.answer
	assert result.plan is not None
	assert len(result.trace) >= 4
	assert any(event.stage == "retrieve" for event in result.trace)
	assert result.retrieved_docs
	history = session_store.get_messages("s1")
	assert len(history) == 2
	assert history[0].role == "user"
	assert history[1].role == "assistant"


def test_agent_can_call_tool_and_record_tool_trace():
	model = FakeToolAwareChatModel(
		[
			AIMessage(
				content="",
				tool_calls=[
					{
						"name": "interview_checklist",
						"args": {"topic": "LangGraph Agent", "seniority": "senior"},
						"id": "call_1",
						"type": "tool_call",
					}
				],
			),
			AIMessage(content="你可以先讲清楚状态图、工具路由和可观测性，再补充 checklist 中的项目案例。"),
		]
	)
	agent, _ = build_agent(model)

	result = agent.respond("帮我准备 LangGraph Agent 面试", session_id="s2")

	assert "状态图" in result.answer
	assert len(result.tool_calls) == 1
	assert result.tool_calls[0].tool_name == "interview_checklist"
	assert "LangGraph Agent" in result.tool_calls[0].tool_output
	assert any(event.stage == "tool" for event in result.trace)


def test_agent_keeps_multi_turn_history_in_same_session():
	model = FakeToolAwareChatModel(
		[
			AIMessage(content="第一次回答"),
			AIMessage(content="第二次回答"),
		]
	)
	agent, _ = build_agent(model)

	first = agent.respond("第一问", session_id="multi")
	second = agent.respond("第二问", session_id="multi")

	assert first.answer == "第一次回答"
	assert second.answer == "第二次回答"
	assert second.trace[0].stage == "session"
	assert "2 historical messages" in second.trace[0].message


def test_agent_stream_exposes_trace_events_in_order():
	model = FakeToolAwareChatModel([AIMessage(content="流式回答")])
	agent, _ = build_agent(model)

	events = list(agent.stream("帮我总结 Agent 面试重点", session_id="stream"))

	assert events
	assert events[0].stage == "session"
	assert any(event.stage == "plan" for event in events)
	assert any(event.stage == "retrieve" for event in events)
	assert events[-1].stage == "finalize"


