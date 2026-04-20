from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title=settings.app_name)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}
