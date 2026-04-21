from collections import defaultdict
from typing import Dict, List


class SessionStore:
    def __init__(self):
        self._store: Dict[str, List[dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str):
        self._store[session_id].append({
            "role": role,
            "content": content
        })

    def get_messages(self, session_id: str) -> List[dict]:
        return self._store.get(session_id, [])

    def get_recent_messages(self, session_id: str, limit: int = 6) -> List[dict]:
        return self._store.get(session_id, [])[-limit:]

    def format_recent_context(self, session_id: str, limit: int = 6) -> str:
        messages = self.get_recent_messages(session_id, limit=limit)
        if not messages:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def clear(self, session_id: str):
        self._store.pop(session_id, None)


session_store = SessionStore()
