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


class PlanRouteResponse(BaseModel):
    template_id: str
    candidate_template_ids: list[str]
    selection_strategy: str
    selection_reason: str
    selection_confidence: float | None = None
    fallback_used: bool
    candidate_details: list[dict[str, object]]
    route_trace: list[dict[str, object]]


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    plan_summary: str | None = None
    plan_route: PlanRouteResponse | None = None
    candidate_details: list[dict[str, object]]
    route_trace: list[dict[str, object]]
    trace: list[dict[str, object]]
    retrieved_docs: list[dict[str, object]]
    tool_calls: list[dict[str, object]]


@lru_cache(maxsize=1)
def get_agent():
    return create_interview_agent()


def serialize_result(result: ExecutionResult) -> ChatResponse:
    candidate_details = [
        {
            "template_id": item.template_id,
            "score": item.score,
            "priority": item.priority,
            "matched_keywords": item.matched_keywords,
            "selected": item.selected,
            "rejected_reason": item.rejected_reason,
        }
        for item in (result.plan.candidate_details if result.plan else [])
    ]
    route_trace = [
        {"stage": item.stage, "message": item.message, "data": item.data}
        for item in (result.plan.route_trace if result.plan else [])
    ]

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        plan_summary=result.plan.reasoning_summary if result.plan else None,
        plan_route=(
            PlanRouteResponse(
                template_id=result.plan.template_id,
                candidate_template_ids=result.plan.candidate_template_ids,
                selection_strategy=result.plan.selection_strategy,
                selection_reason=result.plan.selection_reason,
                selection_confidence=result.plan.selection_confidence,
                fallback_used=result.plan.fallback_used,
                candidate_details=candidate_details,
                route_trace=route_trace,
            )
            if result.plan
            else None
        ),
        candidate_details=candidate_details,
        route_trace=route_trace,
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

