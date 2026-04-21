from fastapi import APIRouter
from app.ingestion.ingest_service import knowledge_ingest_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/ingest")
def ingest_knowledge():
    return knowledge_ingest_service.ingest_markdown_directory()
