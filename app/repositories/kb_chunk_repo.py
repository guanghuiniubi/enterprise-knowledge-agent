import json
from sqlalchemy import delete, text
from app.db.session import SessionLocal
from app.models.kb_chunk import KBChunk


class KBChunkRepo:
    def delete_by_document_id(self, document_id: int):
        with SessionLocal() as db:
            stmt = delete(KBChunk).where(KBChunk.document_id == document_id)
            db.execute(stmt)
            db.commit()

    def bulk_insert(self, rows: list[dict]):
        if not rows:
            return

        with SessionLocal() as db:
            db.bulk_insert_mappings(KBChunk, rows)
            db.commit()

    def search_by_vector(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        query_vector_str = json.dumps(query_vector)

        sql = text(
            """
            select
                c.id,
                c.document_id,
                c.chunk_index,
                c.header_path,
                c.content,
                c.token_count,
                c.metadata,
                d.title,
                d.file_name,
                1 - (c.embedding <=> cast(:query_vector as vector)) as score
            from kb_chunk c
                     join kb_document d on c.document_id = d.id
            where c.embedding is not null
            order by c.embedding <=> cast(:query_vector as vector)
            limit :top_k
            """
        )

        with SessionLocal() as db:
            rows = db.execute(
                sql,
                {
                    "query_vector": query_vector_str,
                    "top_k": top_k
                }
            ).mappings().all()

        return [dict(row) for row in rows]


kb_chunk_repo = KBChunkRepo()
