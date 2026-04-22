from fastapi import APIRouter

from app.memory.session_store import session_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
def list_sessions(limit: int = 30):
    return {
        "sessions": session_store.list_sessions(limit=limit)
    }


@router.get("/{session_id}/messages")
def get_session_messages(session_id: str):
    return {
        "session_id": session_id,
        "messages": session_store.get_messages(session_id)
    }


@router.delete("/{session_id}")
def delete_session(session_id: str):
    session_store.clear(session_id)
    return {"status": "ok", "session_id": session_id}

