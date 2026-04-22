import time

from app.embeddings.local_embedding import local_embedding_service
from app.core.config import settings
from app.repositories.kb_chunk_repo import kb_chunk_repo
from app.observability.metrics import observability_manager
from app.observability.tracing import traceable
from app.rag.reranker import hybrid_reranker
from app.security.access_control import AccessContext, knowledge_access_controller


class VectorKnowledgeRetriever:
    @traceable(name="vector_search")
    def search(self, query: str, top_k: int = 5, access_context: AccessContext | None = None) -> list[dict]:
        started = time.perf_counter()

        query_vector = local_embedding_service.embed_query(query)
        candidate_top_k = max(top_k, top_k * settings.rerank_candidate_multiplier)
        rows = kb_chunk_repo.search_by_vector(query_vector=query_vector, top_k=candidate_top_k)
        accessible_rows = knowledge_access_controller.filter_rows(rows, access_context)

        rerank_started = time.perf_counter()
        ranked_rows = hybrid_reranker.rerank_vector_results(query=query, rows=accessible_rows, top_k=top_k)
        rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 3)

        results = []
        for row in ranked_rows:
            results.append({
                "id": str(row["id"]),
                "document_id": str(row["document_id"]),
                "title": row["title"],
                "content": row["content"],
                "score": float(row["score"]),
                "rerank_score": float(row.get("rerank_score", 0.0)),
                "metadata": row.get("metadata", {}),
            })

        total_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        observability_manager.record_retrieval(
            source="pgvector",
            latency_ms=total_latency_ms,
            candidate_count=len(rows),
            accessible_count=len(accessible_rows),
            result_count=len(results),
        )
        observability_manager.record_rerank(
            strategy="hybrid_vector_keyword_diversity",
            latency_ms=rerank_latency_ms,
            input_count=len(accessible_rows),
            output_count=len(results),
        )

        return results


vector_retriever = VectorKnowledgeRetriever()
