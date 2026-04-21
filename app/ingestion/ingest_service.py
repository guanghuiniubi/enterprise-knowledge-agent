from app.core.config import settings
from app.embeddings.local_embedding import local_embedding_service
from app.ingestion.chunker import markdown_chunker
from app.ingestion.markdown_loader import markdown_loader
from app.repositories.kb_document_repo import kb_document_repo
from app.repositories.kb_chunk_repo import kb_chunk_repo
from app.observability.tracing import traceable


class KnowledgeIngestService:
    @traceable(name="knowledge_ingest")
    def ingest_markdown_directory(self) -> dict:
        docs = markdown_loader.load_directory(settings.knowledge_base_path)

        total_docs = 0
        total_chunks = 0

        for doc in docs:
            doc_record = kb_document_repo.upsert(
                source_path=doc["source_path"],
                file_name=doc["file_name"],
                title=doc["title"],
                metadata_json=doc["metadata"]
            )

            kb_chunk_repo.delete_by_document_id(doc_record.id)

            chunks = markdown_chunker.split(doc["content"], chunk_size=500, overlap=80)
            texts = [chunk["content"] for chunk in chunks]
            embeddings = local_embedding_service.embed_texts(texts) if texts else []

            rows = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                rows.append({
                    "document_id": doc_record.id,
                    "chunk_index": idx,
                    "header_path": chunk["header_path"],
                    "content": chunk["content"],
                    "token_count": chunk["token_count"],
                    "metadata_json": {
                        **doc["metadata"],
                        "header_path": chunk["header_path"],
                        "chunk_local_index": chunk["chunk_local_index"]
                    },
                    "embedding": embedding
                })

            kb_chunk_repo.bulk_insert(rows)

            total_docs += 1
            total_chunks += len(rows)

        return {
            "status": "ok",
            "documents": total_docs,
            "chunks": total_chunks
        }


knowledge_ingest_service = KnowledgeIngestService()
