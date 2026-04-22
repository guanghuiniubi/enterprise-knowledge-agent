from __future__ import annotations

import re
from pathlib import Path

from eka.core.base import BaseRetriever
from eka.core.types import RetrievedDocument


class KeywordKnowledgeBaseRetriever(BaseRetriever):
	"""A simple keyword-overlap retriever over local Markdown/Text files."""

	SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

	def __init__(self, root_dir: Path | str | None) -> None:
		self.root_dir = Path(root_dir).expanduser().resolve() if root_dir else None

	def retrieve(self, query: str, limit: int = 4) -> list[RetrievedDocument]:
		if not self.root_dir or not self.root_dir.exists():
			return []

		tokens = self._tokenize(query)
		results: list[RetrievedDocument] = []
		for file_path in self.root_dir.rglob("*"):
			if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
				continue
			try:
				content = file_path.read_text(encoding="utf-8")
			except UnicodeDecodeError:
				content = file_path.read_text(encoding="utf-8", errors="ignore")

			score = self._score(tokens, content)
			if score <= 0:
				continue
			preview = content.strip().replace("\n", " ")[:600]
			results.append(
				RetrievedDocument(
					source=str(file_path),
					content=preview,
					score=score,
					metadata={"filename": file_path.name},
				)
			)

		results.sort(key=lambda item: item.score, reverse=True)
		return results[:limit]

	def _tokenize(self, text: str) -> set[str]:
		return {token for token in re.split(r"[^\w\u4e00-\u9fff]+", text.lower()) if len(token) > 1}

	def _score(self, query_tokens: set[str], content: str) -> float:
		if not query_tokens:
			return 0.0
		content_lower = content.lower()
		overlap = sum(1 for token in query_tokens if token in content_lower)
		return overlap / len(query_tokens)

