import time
from uuid import uuid4

from app.core.logging import setup_logging
from app.core.langsmith import setup_langsmith

setup_logging()
setup_langsmith()

from fastapi import FastAPI, Request
from app.api.chat import router as chat_router
from app.api.evaluation import router as evaluation_router
from app.api.knowledge import router as knowledge_router
from app.api.prompts import router as prompts_router
from app.api.sessions import router as sessions_router
from app.api.state import router as state_router
from app.api.system import router as system_router
from app.api.ui import router as ui_router
from app.core.config import settings
from app.db.init_db import init_db
from app.core.request_context import clear_request_context, set_request_context
from app.observability.metrics import observability_manager

init_db()

app = FastAPI(title=settings.app_name)


@app.middleware("http")
async def observability_http_middleware(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    set_request_context(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        status_code = response.status_code if response is not None else 500
        observability_manager.record_http_request(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=latency_ms,
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        clear_request_context()


app.include_router(ui_router)
app.include_router(chat_router)
app.include_router(prompts_router)
app.include_router(evaluation_router)
app.include_router(knowledge_router)
app.include_router(sessions_router)
app.include_router(state_router)
app.include_router(system_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
