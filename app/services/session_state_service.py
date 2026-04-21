from app.repositories.session_state_repo import session_state_repo


class SessionStateService:
    def get(self, session_id: str):
        return session_state_repo.get_by_session_id(session_id)

    def save_waiting_state(
            self,
            session_id: str,
            user_id: str,
            current_intent: str,
            pending_slots: list[str],
            collected_slots: dict | None = None
    ):
        return session_state_repo.upsert(
            session_id=session_id,
            user_id=user_id,
            current_intent=current_intent,
            pending_slots=pending_slots,
            collected_slots=collected_slots or {},
            status="waiting_clarification"
        )

    def save_completed_state(
            self,
            session_id: str,
            user_id: str,
            current_intent: str | None,
            collected_slots: dict
    ):
        return session_state_repo.upsert(
            session_id=session_id,
            user_id=user_id,
            current_intent=current_intent,
            pending_slots=[],
            collected_slots=collected_slots,
            status="completed"
        )

    def clear(self, session_id: str):
        session_state_repo.clear(session_id)


session_state_service = SessionStateService()
