from __future__ import annotations

import re
from pathlib import Path

from eka.core.base import BaseRetriever
from eka.core.types import RetrievedDocument


class KeywordKnowledgeBaseRetriever(BaseRetriever):
	"""A simple keyword-overlap retriever over local Markdown/Text files.

	这是当前项目里最轻量的一种检索器实现，适合：
	- 学习 Agent / RAG 的基本流转
	- 在没有向量库时先跑通链路
	- 验证 planner / retriever / executor 的接口是否合理

	它并不是“高质量检索”的最终形态，更像是一个 baseline：
	- 不做 embedding
	- 不做向量召回
	- 不做 rerank
	- 只做关键词重叠匹配

	但它的优点是：
	- 非常容易理解
	- 调试成本低
	- 和后续更强的 retriever 抽象兼容
	"""

	SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

	def __init__(self, root_dir: Path | str | None) -> None:
		# 如果没有提供知识库目录，retriever 会变成“空检索器”，
		# 但整个 Agent 流程依然能跑通，只是 retrieve 阶段拿不到外部知识。
		self.root_dir = Path(root_dir).expanduser().resolve() if root_dir else None

	def retrieve(self, query: str, limit: int = 4) -> list[RetrievedDocument]:
		"""Retrieve top documents using simple keyword overlap.

		主流程可以拆成：
		1. 先把 query 分词
		2. 遍历知识库文件
		3. 计算每个文件和 query 的关键词重叠分数
		4. 返回 score 最高的前 `limit` 个文件

		它的复杂度不低，因为是直接扫目录全文；
		所以真实工程里通常会被向量检索或混合检索替代。
		"""
		if not self.root_dir or not self.root_dir.exists():
			return []

		# 先把 query 归一化成 token 集合。
		tokens = self._tokenize(query)
		results: list[RetrievedDocument] = []
		for file_path in self.root_dir.rglob("*"):
			# 只处理文本类知识文件，避免把二进制或无关文件扫进去。
			if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
				continue
			try:
				content = file_path.read_text(encoding="utf-8")
			except UnicodeDecodeError:
				# 某些文件编码可能不标准，这里用 ignore 保证流程不要因为坏字符中断。
				content = file_path.read_text(encoding="utf-8", errors="ignore")

			score = self._score(tokens, content)
			if score <= 0:
				# 一个关键词都没命中时，直接跳过，避免把无关文档塞进上下文。
				continue
			# 这里只截断一个 preview，而不是返回全文，避免 prompt 被无上限膨胀。
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
		"""Split mixed Chinese/English text into a simple token set.

		这里的 tokenizer 很粗糙，但足够用于 baseline：
		- 英文按非字母数字切分
		- 中文按连续字符片段保留
		- 过滤掉长度为 1 的 token，降低噪声
		"""
		return {token for token in re.split(r"[^\w\u4e00-\u9fff]+", text.lower()) if len(token) > 1}

	def _score(self, query_tokens: set[str], content: str) -> float:
		"""Compute a naive overlap ratio between query tokens and document content.

		分数定义为：
		命中的 query token 数量 / query token 总数

		因此它更像“召回相关度”的粗略估计，而不是严格的语义相似度。
		"""
		if not query_tokens:
			return 0.0
		content_lower = content.lower()
		overlap = sum(1 for token in query_tokens if token in content_lower)
		return overlap / len(query_tokens)
