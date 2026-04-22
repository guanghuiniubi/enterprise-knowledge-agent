from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

INDEX_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"


@router.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=INDEX_PATH.read_text(encoding="utf-8"))

