from __future__ import annotations

from functools import lru_cache

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from eka.agents import create_interview_agent
from eka.core.types import ExecutionResult


class ChatRequest(BaseModel):
    question: str = Field(description="User interview question")
    session_id: str = Field(default="default", description="Conversation session id")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    plan_summary: str | None = None
    trace: list[dict[str, object]]
    retrieved_docs: list[dict[str, object]]
    tool_calls: list[dict[str, object]]


@lru_cache(maxsize=1)
def get_agent():
    return create_interview_agent()


def serialize_result(result: ExecutionResult) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        plan_summary=result.plan.reasoning_summary if result.plan else None,
        trace=[{"stage": item.stage, "message": item.message, "data": item.data} for item in result.trace],
        retrieved_docs=[
            {
                "source": item.source,
                "content": item.content,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in result.retrieved_docs
        ],
        tool_calls=[
            {
                "tool_name": item.tool_name,
                "tool_input": item.tool_input,
                "tool_output": item.tool_output,
            }
            for item in result.tool_calls
        ],
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        result = get_agent().respond(request.question, session_id=request.session_id)
        return serialize_result(result)

    return app


app = create_app()


def main() -> None:
    uvicorn.run("eka.api.app:app", host="127.0.0.1", port=8000, reload=False)

