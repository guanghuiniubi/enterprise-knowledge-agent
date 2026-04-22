from app.embeddings.local_embedding import local_embedding_service
from app.core.config import settings
from app.repositories.kb_chunk_repo import kb_chunk_repo
from app.observability.tracing import traceable
from app.rag.reranker import hybrid_reranker


class VectorKnowledgeRetriever:
    @traceable(name="vector_search")
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = local_embedding_service.embed_query(query)
        candidate_top_k = max(top_k, top_k * settings.rerank_candidate_multiplier)
        rows = kb_chunk_repo.search_by_vector(query_vector=query_vector, top_k=candidate_top_k)

        results = []
        for row in hybrid_reranker.rerank_vector_results(query=query, rows=rows, top_k=top_k):
            results.append({
                "id": str(row["id"]),
                "document_id": str(row["document_id"]),
                "title": row["title"],
                "content": row["content"],
                "score": float(row["score"]),
                "rerank_score": float(row.get("rerank_score", 0.0)),
                "metadata": row.get("metadata", {})
            })

        return results


vector_retriever = VectorKnowledgeRetriever()
