from fastapi import APIRouter
from app.services.session_state_service import session_state_service

router = APIRouter()


@router.get("/session_state/{session_id}")
def get_session_state(session_id: str):
    state = session_state_service.get(session_id)
    if not state:
        return {"found": False}
    return {
        "found": True,
        "session_id": state.session_id,
        "user_id": state.user_id,
        "current_intent": state.current_intent,
        "pending_slots": state.pending_slots,
        "collected_slots": state.collected_slots,
        "status": state.status
    }
