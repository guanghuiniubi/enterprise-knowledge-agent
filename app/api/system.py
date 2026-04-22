from fastapi import APIRouter
from sqlalchemy import text

from app.core.governance import governance_manager
from app.db.session import engine
from app.observability.metrics import observability_manager

router = APIRouter()


@router.get("/db/health")
def db_health():
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    return {"status": "ok"}


@router.get("/governance")
def governance_state():
    return governance_manager.snapshot()


@router.get("/observability/metrics")
def observability_metrics():
    return observability_manager.runtime_snapshot()


@router.get("/observability/overview")
def observability_overview():
    return observability_manager.dashboard_snapshot()


@router.get("/observability/alerts")
def observability_alerts():
    return observability_manager.alert_snapshot()
