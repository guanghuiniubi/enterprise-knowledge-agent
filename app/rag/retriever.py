import json
import re
from pathlib import Path
from typing import Dict, List

from app.core.config import settings


class KnowledgeRetriever:
    def __init__(self, data_path: str | None = None):
        self.data_path = self._resolve_data_path(data_path or settings.knowledge_base_path)
        self.docs = self._load_docs()

    def _resolve_data_path(self, configured_path: str) -> Path:
        candidate = Path(configured_path)
        if candidate.is_file():
            return candidate

        if candidate.is_dir():
            nested_json = candidate / "knowledge_docs.json"
            if nested_json.is_file():
                return nested_json

        project_default = Path(__file__).resolve().parents[2] / "data" / "knowledge_docs.json"
        return project_default

    def _load_docs(self) -> List[Dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        docs: List[Dict] = []
        for item in raw_docs:
            docs.append({
                "id": item["id"],
                "title": item["title"],
                "summary": item.get("summary", ""),
                "tags": item.get("tags", []),
                "key_points": item.get("key_points", []),
                "interview_questions": item.get("interview_questions", []),
                "content": item["content"],
            })
        return docs

    def _tokenize(self, text: str) -> set[str]:
        normalized = text.lower()
        chunks = re.findall(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", normalized)
        tokens: set[str] = set(chunks)

        for chunk in chunks:
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", chunk):
                for size in (2, 3, 4):
                    if len(chunk) >= size:
                        for idx in range(len(chunk) - size + 1):
                            tokens.add(chunk[idx: idx + size])
        if not tokens:
            tokens = {char for char in normalized if char.strip()}
        return tokens

    def list_topics(self) -> List[Dict]:
        return [
            {
                "id": doc["id"],
                "title": doc["title"],
                "summary": doc["summary"],
                "tags": doc["tags"],
            }
            for doc in self.docs
        ]

    def get_by_id(self, doc_id: str) -> Dict | None:
        for doc in self.docs:
            if doc["id"] == doc_id:
                return doc
        return None

    def get_by_topic(self, topic: str) -> Dict | None:
        topic_lower = topic.lower()
        for doc in self.docs:
            haystacks = [doc["title"].lower(), doc["summary"].lower(), " ".join(doc["tags"]).lower()]
            if any(topic_lower in value for value in haystacks):
                return doc
        return None

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_tokens = self._tokenize(query)
        scored = []

        for doc in self.docs:
            searchable_text = " ".join([
                doc["title"],
                doc["summary"],
                " ".join(doc["tags"]),
                " ".join(doc["key_points"]),
                doc["content"],
            ]).lower()
            score = 0
            for token in query_tokens:
                if token in searchable_text:
                    score += max(1, len(token))
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                **doc,
                "score": score,
            }
            for score, doc in scored[:top_k]
        ]
