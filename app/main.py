from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.system import router as system_router
from app.api.state import router as state_router
from app.api.knowledge import router as knowledge_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.langsmith import setup_langsmith
from app.db.init_db import init_db

setup_logging()
setup_langsmith()
init_db()

app = FastAPI(title=settings.app_name)

app.include_router(chat_router)
app.include_router(system_router)
app.include_router(state_router)
app.include_router(knowledge_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
