from __future__ import annotations

from eka.core.base import BaseMemory, BaseSessionStore
from eka.core.types import MessageRecord


class SessionMemory(BaseMemory):
    """Memory backed by a session store."""

    def __init__(self, session_store: BaseSessionStore) -> None:
        self.session_store = session_store

    def load(self, session_id: str) -> list[MessageRecord]:
        return self.session_store.get_messages(session_id)

    def save_turn(self, session_id: str, user_input: str, assistant_output: str) -> None:
        self.session_store.append_messages(
            session_id,
            [
                MessageRecord(role="user", content=user_input),
                MessageRecord(role="assistant", content=assistant_output),
            ],
        )

