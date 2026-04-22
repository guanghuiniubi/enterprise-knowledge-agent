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
	docs = "\n".join(
		f"- {doc.metadata.get('filename', doc.source)} (score={doc.score:.2f})"
		for doc in result.retrieved_docs
	) or "- 无"
	return textwrap.dedent(
		f"""
		[Plan Summary]
		{plan_summary}

		[Retrieved Docs]
		{docs}

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


