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
    """Shared state passed between LangGraph nodes.

    这里可以把它想成“当前这一轮 Agent 执行的状态快照”。
    每个节点都会：
    - 读取当前 state
    - 生成一部分新的 state 更新
    - 再交给 LangGraph 合并回总 state

    关键字段说明：
    - `messages`：模型与工具处理所需的标准消息流
    - `plan`：planner 产出的中间计划
    - `retrieved_docs`：retriever 找到的知识片段
    - `tool_calls`：最终从消息流中反推出的结构化工具调用记录
    - `trace`：每个阶段的可观察事件

    `Annotated[..., add_messages]` / `Annotated[..., operator.add]` 的含义是：
    这些字段不是简单覆盖，而是增量合并。
    这也是 LangGraph 里非常重要的“状态合并”机制。
    """
    session_id: str
    user_input: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: PlanRecord
    retrieved_docs: list[RetrievedDocument]
    final_answer: str
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]


class LangGraphAgentExecutor(BaseExecutor):
    """LangGraph-based executor for the interview assistant workflow.

    这个类是当前项目真正的运行时编排核心。

    如果把：
    - `Planner` 看成“决定这轮怎么做”
    - `Retriever` 看成“查资料”
    - `Tool` 看成“可调用能力”
    - `Memory` 看成“历史对话存储”

    那么 `LangGraphAgentExecutor` 就是把它们组织成一条真正会运行的状态机。
    """

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
        # 这些都是依赖注入进来的组件。
        # 这样设计的好处是：
        # - 易测试
        # - 易替换
        # - 易扩展
        self.chat_model = chat_model
        self.memory = memory
        self.planner = planner
        self.retriever = retriever
        self.tools = tools
        # 项目内定义的 BaseTool 需要先转换为 LangChain 可识别的工具对象，
        # 这样 chat model 才能通过 tool calling 协议请求它们。
        self.langchain_tools = [tool.as_langchain_tool() for tool in tools]

        # ToolNode 是 LangGraph 自带的工具执行节点：
        # 它会读取 AIMessage 里的 tool_calls，执行对应工具，再生成 ToolMessage。
        self.tool_node = ToolNode(self.langchain_tools)
        self.system_prompt = system_prompt or (
            "你是一名 AI Agent 面试助手。你的职责是帮助用户准备 AI Agent 工程师面试。"
            "你可以结合规划摘要、知识检索结果和工具输出来回答。"
            "请优先给出结构化、可执行、面试友好的建议。"
            "不要输出原始 chain-of-thought；只输出简洁的分析结论、行动建议和必要的工具结果整合。"
        )
        # 这里把所有节点和边编译成可执行图。
        self.graph = self._build_graph()

    def invoke(self, user_input: str, session_id: str = "default") -> ExecutionResult:
        """Run the whole graph synchronously and convert final state into `ExecutionResult`."""
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
        """Stream trace events from each node update.

        `stream_mode="updates"` 会让 LangGraph 在每个节点完成后返回该节点的增量更新。
        这里我们只提取 `trace` 字段，供 CLI / 前端做阶段性展示。
        """
        for chunk in self.graph.stream(
            self._initial_state(user_input=user_input, session_id=session_id),
            stream_mode="updates",
        ):
            for node_state in chunk.values():
                for event in node_state.get("trace", []):
                    yield event

    def _build_graph(self):
        """Create the LangGraph state machine.

        整个图的主路径是：

        START -> load_context -> plan -> retrieve -> assistant
                                               assistant -> tools -> assistant
                                               assistant -> finalize -> END

        这个图的关键点有两个：
        1. `assistant` 后不是固定下一步，而是条件分支
        2. `tools -> assistant` 形成一个 loop，让模型能在看到工具结果后继续回答
        """
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
        """Build the minimal state required to start graph execution."""
        return {"session_id": session_id, "user_input": user_input}

    def _load_context(self, state: InterviewGraphState) -> dict[str, Any]:
        """Load session history and append the current user message.

        这一阶段的核心任务是把 Memory 里的历史对话，转换成 LangChain 标准消息对象，
        再和当前用户输入拼起来，形成当前轮的上下文消息流。
        """
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
        """Generate a plan for the current turn.

        这里的 planner 并不是直接输出最终回答，
        而是先输出一个“如何回答这一轮问题”的结构化计划。

        当前项目中，这个 plan 还携带了大量可观察路由信息：
        - 选中的模板
        - 候选模板
        - route strategy
        - route trace
        - recall / rerank 细节
        """
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
        """Retrieve supporting knowledge for the current turn.

        检索优先使用 planner 给出的 search query，
        这意味着 planner 不只是做“分类”，还会影响后续知识获取策略。
        """
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
        """Ask the chat model to either answer directly or request tools.

        这是执行图里最重要的决策节点：
        - 如果模型觉得信息已经足够，就直接给最终回答
        - 如果模型觉得还需要额外能力，就返回 tool_calls

        在 OpenAI-compatible tool calling 协议下，这两个动作都会表现为 `AIMessage`，
        区别在于是否带有 `tool_calls` 字段。
        """
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
        """Decide whether to execute tools or finalize the answer.

        这相当于 LangGraph 里的分支路由函数：
        - 有 tool_calls -> 去 `tools`
        - 没有 tool_calls -> 去 `finalize`
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "finalize"

    def _finalize(self, state: InterviewGraphState) -> dict[str, Any]:
        """Extract final artifacts and persist the user/assistant turn.

        这个阶段会做几件收尾工作：
        1. 从消息流中找最终自然语言回答
        2. 从 AIMessage / ToolMessage 中抽取工具调用记录
        3. 把 user / assistant 对话写回 memory
        4. 生成 finalize 阶段的 trace
        """
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
        """Assemble the messages sent to the chat model.

        当前 prompt 由三部分组成：
        - 固定 system prompt
        - 动态 context prompt（plan / retrieved docs / routing metadata）
        - 真正的对话消息流
        """
        return [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=self._context_prompt(state)),
            *state["messages"],
        ]

    def _context_prompt(self, state: InterviewGraphState) -> str:
        """Build a structured context block for the model.

        这里注入的是“可观察上下文”，不是隐藏推理链。
        这使得模型在回答时能利用：
        - 当前 plan
        - 候选模板与最终路由结果
        - 检索结果
        但不会暴露不必要的原始内部思维链。
        """
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
        """Find the last assistant message that is a real answer, not a tool request."""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return self._coerce_text(message.content)
        return ""

    def _extract_tool_calls(self, messages: list[BaseMessage]) -> list[ToolCallRecord]:
        """Reconstruct structured tool-call records from the message history.

        工具调用信息分散在两类消息里：
        - `AIMessage.tool_calls`：模型说“我要调什么工具、参数是什么”
        - `ToolMessage`：工具真正执行后的输出

        所以这里需要做一次“重新拼装”。
        """
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
        """Convert project-level MessageRecord objects into LangChain message objects."""
        if record.role == "assistant":
            return AIMessage(content=record.content)
        if record.role == "system":
            return SystemMessage(content=record.content)
        return HumanMessage(content=record.content)

    def _coerce_text(self, content: Any) -> str:
        """Normalize model/tool content into plain text for trace, CLI, and storage."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

