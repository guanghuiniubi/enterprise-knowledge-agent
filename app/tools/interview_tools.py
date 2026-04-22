import re

from app.observability.tracing import traceable
from app.rag.reranker import hybrid_reranker
from app.rag.vector_retriever import vector_retriever
from app.repositories.kb_chunk_repo import kb_chunk_repo
from app.repositories.kb_document_repo import kb_document_repo


class InterviewToolkit:
    def __init__(self):
        self.retriever = vector_retriever

    def _extract_tags(self, metadata: dict | None) -> list[str]:
        metadata = metadata or {}
        tags = metadata.get("tags") or metadata.get("keywords") or []
        if isinstance(tags, list):
            return [str(item) for item in tags]
        if isinstance(tags, str):
            return [part.strip() for part in tags.split(",") if part.strip()]
        return []

    def _build_summary(self, content_blocks: list[str]) -> str:
        merged = "\n".join(block.strip() for block in content_blocks if block.strip())
        return merged[:240]

    def _aggregate_chunks(self, ranked_chunks: list, all_chunks: list, max_groups: int = 5) -> list:
        chunk_by_index = {chunk.chunk_index: chunk for chunk in all_chunks}
        selected_indexes: set[int] = set()
        for chunk in ranked_chunks[:max_groups]:
            for neighbor in (chunk.chunk_index - 1, chunk.chunk_index, chunk.chunk_index + 1):
                if neighbor in chunk_by_index:
                    selected_indexes.add(neighbor)
        return [chunk_by_index[index] for index in sorted(selected_indexes)]

    def _extract_key_points(self, chunks: list) -> list[str]:
        points: list[str] = []
        seen = set()
        for chunk in chunks:
            if chunk.header_path and chunk.header_path not in seen:
                points.append(chunk.header_path)
                seen.add(chunk.header_path)
            snippet = chunk.content.strip().splitlines()[0][:48] if chunk.content.strip() else ""
            if snippet and snippet not in seen:
                points.append(snippet)
                seen.add(snippet)
            if len(points) >= 6:
                break
        return points

    def _extract_rerank_diagnostics(self, results: list[dict]) -> list[dict]:
        return [
            {
                "id": item["id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "score": item["score"],
                "rerank_score": item.get("rerank_score", 0.0),
            }
            for item in results[:3]
        ]

    def _get_document(self, doc_id: str = "", topic: str = ""):
        document = None
        if doc_id:
            try:
                document = kb_document_repo.get_by_id(int(doc_id))
            except ValueError:
                document = kb_document_repo.get_by_title_like(doc_id)
        elif topic:
            document = kb_document_repo.get_by_title_like(topic)
        return document

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_topics",
                    "description": "列出当前知识库中的面试主题，适合用户问题太泛时先缩小范围。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "按问题检索最相关的面试知识点主题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "用户要查询的面试问题或关键词"},
                            "top_k": {"type": "integer", "description": "返回主题数", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_topic",
                    "description": "读取某个主题的详细内容、关键点和高频面试追问。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string", "description": "主题 ID"},
                            "topic": {"type": "string", "description": "主题标题或关键词"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_quiz",
                    "description": "基于某个主题生成几道面试追问，用于自测。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string", "description": "主题 ID"},
                            "topic": {"type": "string", "description": "主题标题或关键词"},
                            "count": {"type": "integer", "description": "题目数量", "default": 3}
                        },
                        "required": []
                    }
                }
            }
        ]

    @traceable(name="tool_list_topics")
    def list_topics(self) -> dict:
        records = kb_document_repo.list_documents(limit=50)
        topics = [
            {
                "id": str(item.id),
                "title": item.title,
                "summary": item.metadata_json.get("summary", "") if item.metadata_json else "",
                "tags": self._extract_tags(item.metadata_json),
                "source_file": item.file_name,
            }
            for item in records
        ]
        return {
            "topics": topics,
            "count": len(topics),
        }

    @traceable(name="tool_search_knowledge")
    def search_knowledge(self, query: str, top_k: int = 3) -> dict:
        results = self.retriever.search(query=query, top_k=top_k)
        normalized_results = [
            {
                "id": item["document_id"],
                "chunk_id": item["id"],
                "title": item["title"],
                "summary": item["content"][:240],
                "tags": self._extract_tags(item.get("metadata", {})),
                "score": item["score"],
                "rerank_score": item.get("rerank_score", 0.0),
            }
            for item in results
        ]
        return {
            "query": query,
            "results": normalized_results,
            "rerank": {
                "strategy": "hybrid_vector_keyword_diversity",
                "top_candidates": self._extract_rerank_diagnostics(normalized_results),
            },
        }

    @traceable(name="tool_read_topic")
    def read_topic(self, doc_id: str = "", topic: str = "") -> dict:
        document = self._get_document(doc_id=doc_id, topic=topic)
        if not document:
            return {
                "found": False,
                "message": f"没有找到主题：{doc_id or topic}",
            }

        all_chunks = kb_chunk_repo.list_by_document_id(document.id, limit=50)
        if not all_chunks:
            return {
                "found": True,
                "doc": {
                    "id": str(document.id),
                    "title": document.title,
                    "summary": document.metadata_json.get("summary", "") if document.metadata_json else "",
                    "tags": self._extract_tags(document.metadata_json),
                    "key_points": [],
                    "interview_questions": [],
                    "content": "",
                    "source_file": document.file_name,
                }
            }

        rerank_query = topic or document.title
        ranked_chunks = hybrid_reranker.rerank_chunks(rerank_query, all_chunks)
        aggregated_chunks = self._aggregate_chunks(ranked_chunks, all_chunks)
        contents = [chunk.content for chunk in aggregated_chunks]
        summary = document.metadata_json.get("summary", "") if document.metadata_json else ""
        key_points = self._extract_key_points(ranked_chunks)

        return {
            "found": True,
            "doc": {
                "id": str(document.id),
                "title": document.title,
                "summary": summary or self._build_summary(contents),
                "tags": self._extract_tags(document.metadata_json),
                "key_points": key_points,
                "interview_questions": [],
                "content": "\n\n".join(contents),
                "source_file": document.file_name,
            }
        }

    @traceable(name="tool_generate_quiz")
    def generate_quiz(self, doc_id: str = "", topic: str = "", count: int = 3) -> dict:
        payload = self.read_topic(doc_id=doc_id, topic=topic)
        if not payload.get("found"):
            return {
                "found": False,
                "message": f"没有找到主题：{doc_id or topic}",
            }

        doc = payload["doc"]
        key_points = doc.get("key_points", [])
        questions = [
            f"请你先概括一下《{doc['title']}》这个主题的核心目标是什么？",
            f"如果让你结合当前项目解释《{doc['title']}》，你会怎么描述关键流程？",
            f"《{doc['title']}》里最容易被问到的 trade-off 或风险点是什么？",
        ]
        for point in key_points:
            questions.append(f"围绕“{point}”这一部分，面试官可能会继续追问什么？")

        return {
            "found": True,
            "title": doc["title"],
            "questions": questions[:count],
            "count": min(count, len(questions)),
        }


interview_toolkit = InterviewToolkit()

