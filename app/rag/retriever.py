import json
from pathlib import Path
from typing import List, Dict


class KnowledgeRetriever:
    def __init__(self, data_path: str = "data/knowledge_docs.json"):
        self.data_path = Path(data_path)
        self.docs = self._load_docs()

    def _load_docs(self) -> List[Dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        query_tokens = [token for token in query if token.strip()]
        scored = []

        for doc in self.docs:
            score = sum(1 for token in query_tokens if token in doc["content"] or token in doc["title"])
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
