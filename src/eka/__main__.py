from __future__ import annotations

import argparse
import sys
import textwrap

from eka.agents import create_interview_agent


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Interview assistant demo powered by LangGraph.")
	parser.add_argument("--question", help="Ask a single question and exit.")
	parser.add_argument("--session-id", default="demo", help="Conversation session identifier.")
	return parser


def format_trace(trace: list[object]) -> str:
	lines = ["\n[Trace]"]
	for item in trace:
		stage = getattr(item, "stage", "unknown")
		message = getattr(item, "message", str(item))
		lines.append(f"- {stage}: {message}")
	return "\n".join(lines)


def render_answer(result) -> str:
	plan_summary = result.plan.reasoning_summary if result.plan else "无"
	candidate_details = (
		"\n".join(
			f"- {item.template_id}: score={item.score}, keywords={', '.join(item.matched_keywords) if item.matched_keywords else '无'}, selected={item.selected}, rejected_reason={item.rejected_reason or '无'}"
			for item in result.plan.candidate_details
		)
		if result.plan and result.plan.candidate_details
		else "- 无"
	)
	route_trace_details = (
		"\n".join(
			f"- {item.stage}: {item.message}"
			for item in result.plan.route_trace
		)
		if result.plan and result.plan.route_trace
		else "- 无"
	)
	route_details = (
		"\n".join(
			[
				f"- Template: {result.plan.template_id}",
				f"- Candidates: {', '.join(result.plan.candidate_template_ids) if result.plan.candidate_template_ids else '无'}",
				f"- Strategy: {result.plan.selection_strategy}",
				f"- Confidence: {result.plan.selection_confidence if result.plan.selection_confidence is not None else '无'}",
				f"- Fallback: {result.plan.fallback_used}",
				f"- Reason: {result.plan.selection_reason or '无'}",
			]
		)
		if result.plan
		else "- 无"
	)
	docs = "\n".join(
		f"- {doc.metadata.get('filename', doc.source)} (score={doc.score:.2f})"
		for doc in result.retrieved_docs
	) or "- 无"
	tool_calls = "\n".join(
		f"- {item.tool_name}: input={item.tool_input}"
		for item in result.tool_calls
	) or "- 无"
	return textwrap.dedent(
		f"""
		[Plan Summary]
		{plan_summary}

		[Plan Route]
		{route_details}

		[Plan Candidates]
		{candidate_details}

		[Plan Route Trace]
		{route_trace_details}

		[Retrieved Docs]
		{docs}

		[Tool Calls]
		{tool_calls}

		[Answer]
		{result.answer}
		"""
	).strip() + format_trace(result.trace)


def interactive_loop() -> int:
	agent = create_interview_agent()
	session_id = "demo"
	print("Interview assistant is ready. Type 'exit' to quit.")
	while True:
		try:
			question = input("\nYou> ").strip()
		except EOFError:
			print()
			return 0
		if not question:
			continue
		if question.lower() in {"exit", "quit"}:
			return 0
		result = agent.respond(question, session_id=session_id)
		print(render_answer(result))
	return 0


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	agent = create_interview_agent()

	if args.question:
		result = agent.respond(args.question, session_id=args.session_id)
		print(render_answer(result))
		return 0

	return interactive_loop()


if __name__ == "__main__":
	sys.exit(main())


