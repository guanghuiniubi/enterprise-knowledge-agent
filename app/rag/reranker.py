import re
from collections.abc import Iterable
from typing import Any

from app.core.config import settings


class HybridReranker:
    def _tokenize(self, text: str) -> set[str]:
        normalized = (text or "").lower()
        tokens = set(re.findall(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", normalized))
        if not tokens:
            tokens = {char for char in normalized if char.strip()}
        return tokens

    def _normalize_score(self, value: Any) -> float:
        if value is None:
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(score, 1.0))

    def _keyword_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        haystack = (text or "").lower()
        matched = sum(1 for token in query_tokens if token in haystack)
        return matched / max(1, len(query_tokens))

    def _feature_score(
        self,
        *,
        query: str,
        title: str,
        content: str,
        metadata: dict | None,
        score: Any,
        position: int,
    ) -> float:
        metadata = metadata or {}
        query_tokens = self._tokenize(query)
        title_lower = (title or "").lower()
        content_lower = (content or "").lower()
        tags = metadata.get("tags") or metadata.get("keywords") or []
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]
        tag_text = " ".join(str(item) for item in tags).lower()

        vector_score = self._normalize_score(score)
        title_score = self._keyword_overlap(query_tokens, title_lower)
        keyword_score = self._keyword_overlap(query_tokens, content_lower)
        metadata_score = self._keyword_overlap(query_tokens, tag_text)
        if query.lower() and query.lower() in title_lower:
            title_score = min(1.0, title_score + 0.4)
        position_score = max(0.0, 1.0 - position * 0.08)

        return (
            vector_score * settings.rerank_weight_vector
            + title_score * settings.rerank_weight_title
            + keyword_score * settings.rerank_weight_keyword
            + metadata_score * settings.rerank_weight_metadata
            + position_score * settings.rerank_weight_position
        )

    def _diversity_penalty(self, candidate_text: str, selected_texts: Iterable[str]) -> float:
        candidate_tokens = self._tokenize(candidate_text)
        if not candidate_tokens:
            return 0.0
        max_overlap = 0.0
        for text in selected_texts:
            existing_tokens = self._tokenize(text)
            if not existing_tokens:
                continue
            overlap = len(candidate_tokens & existing_tokens) / max(1, len(candidate_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        return max_overlap * (1 - settings.rerank_diversity_lambda)

    def rerank_vector_results(self, query: str, rows: list[dict], top_k: int) -> list[dict]:
        enriched: list[tuple[float, dict]] = []
        for index, row in enumerate(rows):
            final_score = self._feature_score(
                query=query,
                title=row.get("title", ""),
                content=row.get("content", ""),
                metadata=row.get("metadata") or {},
                score=row.get("score"),
                position=index,
            )
            enriched.append((final_score, {**row, "rerank_score": round(final_score, 6)}))

        enriched.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict] = []
        selected_texts: list[str] = []
        for score, row in enriched:
            candidate_text = f"{row.get('title', '')}\n{row.get('content', '')}"
            adjusted_score = score - self._diversity_penalty(candidate_text, selected_texts)
            row["rerank_score"] = round(adjusted_score, 6)
            selected.append(row)
            selected_texts.append(candidate_text)
            selected.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
            selected = selected[:top_k]
            selected_texts = [f"{item.get('title', '')}\n{item.get('content', '')}" for item in selected]
        return selected[:top_k]

    def rerank_chunks(self, query: str, chunks: list) -> list:
        scored: list[tuple[float, Any]] = []
        for index, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata_json", {}) or {}
            header = getattr(chunk, "header_path", "") or ""
            content = getattr(chunk, "content", "") or ""
            raw_score = self._feature_score(
                query=query,
                title=header,
                content=content,
                metadata=metadata,
                score=metadata.get("score", 0.0),
                position=index,
            )
            raw_score += max(0.0, 1.0 - getattr(chunk, "chunk_index", index) * 0.02) * 0.05
            scored.append((raw_score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]


hybrid_reranker = HybridReranker()


