from sqlalchemy import and_, delete, func, select

from app.db.session import SessionLocal
from app.models.session_message import SessionMessage


class SessionStore:
    def add_message(self, session_id: str, role: str, content: str, user_id: str = "system", metadata: dict | None = None):
        with SessionLocal() as db:
            db.add(SessionMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                metadata_json=metadata or {},
            ))
            db.commit()

    def get_messages(self, session_id: str) -> list[dict]:
        with SessionLocal() as db:
            stmt = (
                select(SessionMessage)
                .where(SessionMessage.session_id == session_id)
                .order_by(SessionMessage.created_at.asc(), SessionMessage.id.asc())
            )
            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "role": row.role,
                    "content": row.content,
                    "metadata": row.metadata_json,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def get_recent_messages(self, session_id: str, limit: int = 6) -> list[dict]:
        with SessionLocal() as db:
            stmt = (
                select(SessionMessage)
                .where(SessionMessage.session_id == session_id)
                .order_by(SessionMessage.created_at.desc(), SessionMessage.id.desc())
                .limit(limit)
            )
            rows = list(db.execute(stmt).scalars().all())
            rows.reverse()
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "role": row.role,
                    "content": row.content,
                    "metadata": row.metadata_json,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def format_recent_context(self, session_id: str, limit: int = 6) -> str:
        messages = self.get_recent_messages(session_id, limit=limit)
        if not messages:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def list_sessions(self, limit: int = 30) -> list[dict]:
        with SessionLocal() as db:
            latest_subquery = (
                select(
                    SessionMessage.session_id.label("session_id"),
                    func.max(SessionMessage.created_at).label("last_message_at")
                )
                .group_by(SessionMessage.session_id)
                .subquery()
            )
            stmt = (
                select(SessionMessage)
                .join(
                    latest_subquery,
                    and_(
                        SessionMessage.session_id == latest_subquery.c.session_id,
                        SessionMessage.created_at == latest_subquery.c.last_message_at,
                    )
                )
                .order_by(SessionMessage.created_at.desc(), SessionMessage.id.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
            sessions: list[dict] = []
            for row in rows:
                count_stmt = select(func.count(SessionMessage.id)).where(SessionMessage.session_id == row.session_id)
                message_count = db.execute(count_stmt).scalar_one()
                sessions.append({
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "last_message": row.content[:120],
                    "last_role": row.role,
                    "last_message_at": row.created_at.isoformat() if row.created_at else None,
                    "message_count": int(message_count),
                })
            return sessions

    def clear(self, session_id: str):
        with SessionLocal() as db:
            db.execute(delete(SessionMessage).where(SessionMessage.session_id == session_id))
            db.commit()


session_store = SessionStore()
