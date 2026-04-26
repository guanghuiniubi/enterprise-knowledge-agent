from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from eka.core.base import BaseExecutor, BaseMemory, BasePlanner, BaseRetriever, BaseTool
from eka.core.types import ExecutionResult, MessageRecord, PlanRecord, RetrievedDocument, ToolCallRecord, TraceEvent


class InterviewGraphState(TypedDict, total=False):
    session_id: str
    user_input: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: PlanRecord
    retrieved_docs: list[RetrievedDocument]
    final_answer: str
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]


class LangGraphAgentExecutor(BaseExecutor):
    """LangGraph-based executor for the interview assistant workflow."""

    def __init__(
        self,
        *,
        chat_model: Any,
        memory: BaseMemory,
        planner: BasePlanner,
        retriever: BaseRetriever,
        tools: list[BaseTool],
        system_prompt: str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.memory = memory
        self.planner = planner
        self.retriever = retriever
        self.tools = tools
        self.langchain_tools = [tool.as_langchain_tool() for tool in tools]
        self.tool_node = ToolNode(self.langchain_tools)
        self.system_prompt = system_prompt or (
            "你是一名 AI Agent 面试助手。你的职责是帮助用户准备 AI Agent 工程师面试。"
            "你可以结合规划摘要、知识检索结果和工具输出来回答。"
            "请优先给出结构化、可执行、面试友好的建议。"
            "不要输出原始 chain-of-thought；只输出简洁的分析结论、行动建议和必要的工具结果整合。"
        )
        self.graph = self._build_graph()

    def invoke(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        state = self.graph.invoke(self._initial_state(user_input=user_input, session_id=session_id))
        return ExecutionResult(
            session_id=session_id,
            answer=state.get("final_answer", ""),
            plan=state.get("plan"),
            retrieved_docs=state.get("retrieved_docs", []),
            tool_calls=state.get("tool_calls", []),
            trace=state.get("trace", []),
        )

    def stream(self, user_input: str, session_id: str = "default"):
        for chunk in self.graph.stream(
            self._initial_state(user_input=user_input, session_id=session_id),
            stream_mode="updates",
        ):
            for node_state in chunk.values():
                for event in node_state.get("trace", []):
                    yield event

    def _build_graph(self):
        builder = StateGraph(cast(Any, InterviewGraphState))
        builder.add_node("load_context", cast(Any, self._load_context))
        builder.add_node("plan", cast(Any, self._plan))
        builder.add_node("retrieve", cast(Any, self._retrieve))
        builder.add_node("assistant", cast(Any, self._assistant))
        builder.add_node("tools", self.tool_node)
        builder.add_node("finalize", cast(Any, self._finalize))

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "assistant")
        builder.add_conditional_edges(
            "assistant",
            self._route_assistant,
            {
                "tools": "tools",
                "finalize": "finalize",
            },
        )
        builder.add_edge("tools", "assistant")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _initial_state(self, *, user_input: str, session_id: str) -> InterviewGraphState:
        return {"session_id": session_id, "user_input": user_input}

    def _load_context(self, state: InterviewGraphState) -> dict[str, Any]:
        history = self.memory.load(state["session_id"])
        history_messages = [self._to_langchain_message(record) for record in history]
        return {
            "messages": history_messages + [HumanMessage(content=state["user_input"])],
            "trace": [
                TraceEvent(
                    stage="session",
                    message=f"Loaded {len(history)} historical messages for session '{state['session_id']}'.",
                )
            ],
        }

    def _plan(self, state: InterviewGraphState) -> dict[str, Any]:
        history = self.memory.load(state["session_id"])
        plan = self.planner.create_plan(state["user_input"], history)
        return {
            "plan": plan,
            "trace": [
                TraceEvent(
                    stage="plan",
                    message=plan.reasoning_summary,
                    data={
                        "objective": plan.objective,
                        "template_id": plan.template_id,
                        "candidate_template_ids": plan.candidate_template_ids,
                        "candidate_details": [
                            {
                                "template_id": item.template_id,
                                "score": item.score,
                                "priority": item.priority,
                                "matched_keywords": item.matched_keywords,
                                "selected": item.selected,
                                "rejected_reason": item.rejected_reason,
                            }
                            for item in plan.candidate_details
                        ],
                        "selection_strategy": plan.selection_strategy,
                        "selection_reason": plan.selection_reason,
                        "selection_confidence": plan.selection_confidence,
                        "fallback_used": plan.fallback_used,
                        "route_trace": [
                            {"stage": item.stage, "message": item.message, "data": item.data}
                            for item in plan.route_trace
                        ],
                        "search_queries": plan.search_queries,
                        "tools_to_consider": plan.tools_to_consider,
                    },
                )
            ],
        }

    def _retrieve(self, state: InterviewGraphState) -> dict[str, Any]:
        query = state.get("plan").search_queries[0] if state.get("plan") and state["plan"].search_queries else state["user_input"]
        docs = self.retriever.retrieve(query, limit=4)
        return {
            "retrieved_docs": docs,
            "trace": [
                TraceEvent(
                    stage="retrieve",
                    message=f"Retrieved {len(docs)} knowledge documents.",
                    data={"sources": [doc.source for doc in docs]},
                )
            ],
        }

    def _assistant(self, state: InterviewGraphState) -> dict[str, Any]:
        bound_model = self.chat_model.bind_tools(self.langchain_tools) if self.langchain_tools else self.chat_model
        prompt_messages = self._build_prompt_messages(state)
        response = bound_model.invoke(prompt_messages)
        tool_calls = getattr(response, "tool_calls", [])
        tool_names = [call.get("name", "unknown") for call in tool_calls]
        if tool_names:
            message = f"Model decided to call tools: {', '.join(tool_names)}."
            data = {"tool_calls": tool_calls}
        else:
            message = "Model produced a final answer without additional tool calls."
            data = {"tool_calls": [], "response_preview": self._coerce_text(response.content)[:300]}
        return {
            "messages": [response],
            "trace": [TraceEvent(stage="assistant", message=message, data=data)],
        }

    def _route_assistant(self, state: InterviewGraphState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "finalize"

    def _finalize(self, state: InterviewGraphState) -> dict[str, Any]:
        answer = self._extract_final_answer(state["messages"])
        tool_calls = self._extract_tool_calls(state["messages"])
        self.memory.save_turn(state["session_id"], state["user_input"], answer)

        tool_trace = [
            TraceEvent(
                stage="tool",
                message=f"Tool {item.tool_name} executed successfully.",
                data={"tool_input": item.tool_input, "tool_output": item.tool_output},
            )
            for item in tool_calls
        ]
        tool_trace.append(
            TraceEvent(
                stage="finalize",
                message="Assistant answer saved to session memory.",
                data={"answer_length": len(answer)},
            )
        )
        return {
            "final_answer": answer,
            "tool_calls": tool_calls,
            "trace": tool_trace,
        }

    def _build_prompt_messages(self, state: InterviewGraphState) -> list[BaseMessage]:
        return [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=self._context_prompt(state)),
            *state["messages"],
        ]

    def _context_prompt(self, state: InterviewGraphState) -> str:
        plan = state.get("plan")
        retrieved_docs = state.get("retrieved_docs", [])
        lines = ["以下是本轮回答的可观察上下文，请按需使用："]
        if plan:
            lines.extend(
                [
                    f"- Plan objective: {plan.objective}",
                    f"- Plan summary: {plan.reasoning_summary}",
                    f"- Plan template: {plan.template_id}",
                    f"- Plan candidates: {', '.join(plan.candidate_template_ids) if plan.candidate_template_ids else 'None'}",
                    f"- Plan route strategy: {plan.selection_strategy}",
                    f"- Plan route reason: {plan.selection_reason or 'None'}",
                    f"- Plan route confidence: {plan.selection_confidence if plan.selection_confidence is not None else 'None'}",
                    f"- Plan fallback used: {plan.fallback_used}",
                    f"- Suggested tools: {', '.join(plan.tools_to_consider) if plan.tools_to_consider else 'None'}",
                    f"- Search queries: {', '.join(plan.search_queries) if plan.search_queries else 'None'}",
                ]
            )
        if retrieved_docs:
            lines.append("- Retrieved knowledge snippets:")
            for index, doc in enumerate(retrieved_docs, start=1):
                lines.append(f"  {index}. [{doc.metadata.get('filename', doc.source)}] {doc.content}")
        else:
            lines.append("- No external knowledge matched; rely on interview best practices and tools when helpful.")
        lines.append("请输出最终面试建议时，优先使用中文、结构化表达，并适度引用工具结果。")
        return "\n".join(lines)

    def _extract_final_answer(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return self._coerce_text(message.content)
        return ""

    def _extract_tool_calls(self, messages: list[BaseMessage]) -> list[ToolCallRecord]:
        tool_inputs_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    call_id = call.get("id")
                    if call_id:
                        tool_inputs_by_id[call_id] = (call["name"], call.get("args", {}))

        records: list[ToolCallRecord] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                tool_name, tool_input = tool_inputs_by_id.get(message.tool_call_id, (getattr(message, "name", "unknown"), {}))
                records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=self._coerce_text(message.content),
                    )
                )
        return records

    def _to_langchain_message(self, record: MessageRecord) -> BaseMessage:
        if record.role == "assistant":
            return AIMessage(content=record.content)
        if record.role == "system":
            return SystemMessage(content=record.content)
        return HumanMessage(content=record.content)

    def _coerce_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

