from fastapi import APIRouter
from sqlalchemy import text
from app.core.governance import governance_manager
from app.db.session import engine

router = APIRouter()


@router.get("/db/health")
def db_health():
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    return {"status": "ok"}


@router.get("/governance")
def governance_state():
    return governance_manager.snapshot()

