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

    def clear(self, session_id: str):
        self._store.pop(session_id, None)


session_store = SessionStore()
