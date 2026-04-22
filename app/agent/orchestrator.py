import json

from app.core.governance import GovernanceError, governance_manager
from app.llm.client import llm_client
from app.observability.tracing import traceable
from app.prompts.registry import prompt_registry
from app.schemas.agent import AgentResult
from app.tools.interview_tools import interview_toolkit


class AgentOrchestrator:
    MAX_STEPS = 6
    MAX_TOOL_RETRIES = 2

    def __init__(self):
        self.toolkit = interview_toolkit

    def _system_prompt(self) -> str:
        return prompt_registry.render("agent_system")

    def _build_messages(self, question: str, context: str = "") -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": prompt_registry.render(
                    "agent_user",
                    context=context or "无",
                    question=question,
                ),
            }
        ]

    def _dispatch_tool(self, name: str, arguments: dict) -> dict:
        if name == "list_topics":
            return self.toolkit.list_topics()
        if name == "search_knowledge":
            return self.toolkit.search_knowledge(
                query=arguments.get("query", ""),
                top_k=int(arguments.get("top_k", 3))
            )
        if name == "read_topic":
            return self.toolkit.read_topic(
                doc_id=arguments.get("doc_id", ""),
                topic=arguments.get("topic", "")
            )
        if name == "generate_quiz":
            return self.toolkit.generate_quiz(
                doc_id=arguments.get("doc_id", ""),
                topic=arguments.get("topic", ""),
                count=int(arguments.get("count", 3))
            )
        return {"error": f"unknown tool: {name}"}

    def _parse_tool_arguments(self, raw_arguments: str) -> tuple[dict, str | None]:
        try:
            return json.loads(raw_arguments or "{}"), None
        except json.JSONDecodeError as exc:
            return {}, f"invalid tool arguments: {exc}"

    def _execute_tool_with_retry(self, name: str, arguments: dict) -> tuple[dict, list[dict]]:
        attempts: list[dict] = []
        for attempt in range(1, self.MAX_TOOL_RETRIES + 2):
            try:
                observation = governance_manager.execute_tool(
                    name,
                    lambda: self._dispatch_tool(name=name, arguments=arguments),
                )
                attempts.append({"attempt": attempt, "success": True})
                if attempt > 1 and isinstance(observation, dict):
                    observation = {
                        **observation,
                        "retry_recovered": True,
                        "attempts": attempt,
                    }
                return observation, attempts
            except Exception as exc:  # noqa: BLE001
                attempts.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(exc),
                })

        return {
            "error": f"tool {name} failed after retries",
            "recoverable": False,
            "attempts": attempts,
        }, attempts

    def _append_citations(self, citations: list[dict], tool_name: str, observation: dict):
        if tool_name == "read_topic" and observation.get("found"):
            doc = observation["doc"]
            citations.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "snippet": doc["summary"] or doc["content"][:120],
            })

    def _build_result(
            self,
            *,
            answer: str,
            route: str,
            tool_calls: list[dict],
            agent_steps: list[dict],
            citations: list[dict],
            need_clarification: bool = False,
            clarification_question: str | None = None,
            fallback: bool = False,
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            route=route,
            tool_calls=tool_calls,
            agent_steps=agent_steps,
            citations=citations,
            need_clarification=need_clarification,
            clarification_question=clarification_question,
            debug={
                "mode": "tool-calling-agent",
                "steps": len(agent_steps),
                "fallback": fallback,
                "prompt_versions": prompt_registry.active_versions(),
                "governance": governance_manager.snapshot(),
            }
        )

    def _fallback_answer(self, question: str, context: str, agent_steps: list[dict]) -> str:
        observations = [
            json.dumps(step.get("observation", {}), ensure_ascii=False)
            for step in agent_steps
            if step.get("observation")
        ]
        try:
            return llm_client.chat(
                system_prompt=prompt_registry.render("fallback_summary_system"),
                user_prompt=prompt_registry.render(
                    "fallback_summary_user",
                    context=context or "无",
                    question=question,
                    observations=chr(10).join(observations) or "无",
                ),
            )
        except GovernanceError as exc:
            return self._deterministic_degraded_answer(question=question, agent_steps=agent_steps, reason=str(exc))

    def _deterministic_degraded_answer(self, question: str, agent_steps: list[dict], reason: str) -> str:
        observed_points: list[str] = []
        for step in agent_steps:
            observation = step.get("observation") or {}
            if observation.get("results"):
                titles = [item.get("title", "") for item in observation["results"][:2] if item.get("title")]
                if titles:
                    observed_points.append(f"已检索到相关主题：{'、'.join(titles)}")
            if observation.get("doc"):
                doc = observation["doc"]
                title = doc.get("title", "相关主题")
                summary = doc.get("summary") or (doc.get("content") or "")[:120]
                observed_points.append(f"已读取《{title}》：{summary}")

        core = "；".join(observed_points[:3]) or "当前没有拿到足够的外部知识结果。"
        return (
            f"核心结论：当前系统触发了降级治理，原因是：{reason}。\n"
            f"回答思路：先基于已经拿到的观察结果给出保守结论，再建议稍后重试或缩小问题范围。\n"
            f"已知信息：{core}\n"
            f"易错点：在模型或工具超时/熔断时，不要把未验证信息当作最终事实。\n"
            f"可追问：你可以把问题改成更具体的子问题，例如“{question} 的核心概念是什么？”。"
        )

    def _run_core(self, question: str, context: str = ""):
        messages = self._build_messages(question=question, context=context)
        tool_calls: list[dict] = []
        agent_steps: list[dict] = []
        citations: list[dict] = []
        tools = self.toolkit.tool_schemas()

        for step_index in range(1, self.MAX_STEPS + 1):
            try:
                llm_result = llm_client.chat_with_tools(messages=messages, tools=tools)
            except GovernanceError as exc:
                result = self._build_result(
                    answer=self._deterministic_degraded_answer(question=question, agent_steps=agent_steps, reason=str(exc)),
                    route="agent_answer",
                    tool_calls=tool_calls,
                    agent_steps=agent_steps,
                    citations=citations,
                    fallback=True,
                )
                yield {"type": "final", "result": result}
                return
            planning_note = llm_result.get("content", "").strip()
            pending_tool_calls = llm_result.get("tool_calls", []) or []

            if pending_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": planning_note,
                    "tool_calls": [
                        {
                            "id": item["id"],
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": item["arguments"],
                            }
                        }
                        for item in pending_tool_calls
                    ]
                }
                messages.append(assistant_message)

                for item in pending_tool_calls:
                    arguments, parse_error = self._parse_tool_arguments(item["arguments"])
                    if parse_error:
                        observation = {
                            "error": parse_error,
                            "recoverable": True,
                            "raw_arguments": item["arguments"],
                        }
                        attempts = [{"attempt": 1, "success": False, "error": parse_error}]
                    else:
                        observation, attempts = self._execute_tool_with_retry(name=item["name"], arguments=arguments)

                    tool_call = {
                        "tool_name": item["name"],
                        "tool_input": arguments,
                        "tool_output": observation,
                    }
                    tool_calls.append(tool_call)
                    self._append_citations(citations, item["name"], observation)

                    step_payload = {
                        "step": step_index,
                        "thought": planning_note or f"调用工具 {item['name']}",
                        "action": item["name"],
                        "action_input": arguments,
                        "observation": observation,
                        "attempts": attempts,
                    }
                    agent_steps.append(step_payload)
                    yield {"type": "tool_result", **step_payload}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": item["id"],
                        "name": item["name"],
                        "content": json.dumps(observation, ensure_ascii=False),
                    })
                continue

            final_text = planning_note or ""
            if final_text.startswith("CLARIFICATION:"):
                clarification_question = final_text.split(":", 1)[1].strip() or "你想重点准备哪一类面试知识点？"
                result = self._build_result(
                    answer=clarification_question,
                    route="clarification",
                    tool_calls=tool_calls,
                    agent_steps=agent_steps + [{
                        "step": step_index,
                        "thought": "问题过于宽泛，需要先澄清范围",
                        "action": "clarification",
                    }],
                    citations=citations,
                    need_clarification=True,
                    clarification_question=clarification_question,
                )
                yield {"type": "final", "result": result}
                return

            result = self._build_result(
                answer=final_text or self._fallback_answer(question, context, agent_steps),
                route="agent_answer",
                tool_calls=tool_calls,
                agent_steps=agent_steps + [{
                    "step": step_index,
                    "thought": planning_note or "信息已足够，生成最终答案",
                    "action": "final_answer",
                }],
                citations=citations,
            )
            yield {"type": "final", "result": result}
            return

        result = self._build_result(
            answer=self._fallback_answer(question, context, agent_steps),
            route="agent_answer",
            tool_calls=tool_calls,
            agent_steps=agent_steps,
            citations=citations,
            fallback=True,
        )
        yield {"type": "final", "result": result}

    @traceable(name="agent_orchestrator_run")
    def run(self, question: str, context: str = "") -> AgentResult:
        for event in self._run_core(question=question, context=context):
            if event["type"] == "final":
                return event["result"]
        raise RuntimeError("Agent did not produce a final result")

    @traceable(name="agent_orchestrator_stream")
    def run_stream(self, question: str, context: str = ""):
        yield {"type": "start", "question": question}
        yield from self._run_core(question=question, context=context)


agent_orchestrator = AgentOrchestrator()
