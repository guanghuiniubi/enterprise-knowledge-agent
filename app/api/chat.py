import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.governance import CircuitBreakerOpen, ExecutionTimeout, RateLimitExceeded
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return chat_service.chat(req)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ExecutionTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except CircuitBreakerOpen as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_stream():
        try:
            yield from chat_service.chat_stream(req)
        except RateLimitExceeded as exc:
            yield f"data: {json.dumps({'type': 'error', 'status': 429, 'detail': str(exc)}, ensure_ascii=False)}\n\n"
        except ExecutionTimeout as exc:
            yield f"data: {json.dumps({'type': 'error', 'status': 504, 'detail': str(exc)}, ensure_ascii=False)}\n\n"
        except CircuitBreakerOpen as exc:
            yield f"data: {json.dumps({'type': 'error', 'status': 503, 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )

