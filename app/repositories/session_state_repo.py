from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.session_state import AgentSessionState


class SessionStateRepo:
    def get_by_session_id(self, session_id: str) -> AgentSessionState | None:
        with SessionLocal() as db:
            stmt = select(AgentSessionState).where(AgentSessionState.session_id == session_id)
            return db.execute(stmt).scalar_one_or_none()

    def upsert(
            self,
            session_id: str,
            user_id: str,
            current_intent: str | None,
            pending_slots: list,
            collected_slots: dict,
            status: str
    ) -> AgentSessionState:
        with SessionLocal() as db:
            stmt = select(AgentSessionState).where(AgentSessionState.session_id == session_id)
            record = db.execute(stmt).scalar_one_or_none()

            if record is None:
                record = AgentSessionState(
                    session_id=session_id,
                    user_id=user_id,
                    current_intent=current_intent,
                    pending_slots=pending_slots,
                    collected_slots=collected_slots,
                    status=status
                )
                db.add(record)
            else:
                record.user_id = user_id
                record.current_intent = current_intent
                record.pending_slots = pending_slots
                record.collected_slots = collected_slots
                record.status = status

            db.commit()
            db.refresh(record)
            return record

    def clear(self, session_id: str):
        with SessionLocal() as db:
            stmt = select(AgentSessionState).where(AgentSessionState.session_id == session_id)
            record = db.execute(stmt).scalar_one_or_none()
            if record:
                record.current_intent = None
                record.pending_slots = []
                record.collected_slots = {}
                record.status = "completed"
                db.commit()


session_state_repo = SessionStateRepo()
