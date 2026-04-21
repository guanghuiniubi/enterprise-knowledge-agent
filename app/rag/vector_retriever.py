from app.embeddings.local_embedding import local_embedding_service
from app.repositories.kb_chunk_repo import kb_chunk_repo
from app.observability.tracing import traceable


class VectorKnowledgeRetriever:
    @traceable(name="vector_search")
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = local_embedding_service.embed_query(query)
        rows = kb_chunk_repo.search_by_vector(query_vector=query_vector, top_k=top_k)

        results = []
        for row in rows:
            results.append({
                "id": str(row["id"]),
                "title": row["title"],
                "content": row["content"],
                "score": float(row["score"]),
                "metadata": row["metadata"]
            })

        return results


vector_retriever = VectorKnowledgeRetriever()
