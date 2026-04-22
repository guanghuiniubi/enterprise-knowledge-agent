from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.kb_document import KBDocument


class KBDocumentRepo:
    def list_documents(self, limit: int = 50) -> list[KBDocument]:
        with SessionLocal() as db:
            stmt = select(KBDocument).order_by(KBDocument.created_at.desc()).limit(limit)
            return list(db.execute(stmt).scalars().all())

    def get_by_id(self, document_id: int) -> KBDocument | None:
        with SessionLocal() as db:
            stmt = select(KBDocument).where(KBDocument.id == document_id)
            return db.execute(stmt).scalar_one_or_none()

    def get_by_title_like(self, query: str) -> KBDocument | None:
        with SessionLocal() as db:
            stmt = select(KBDocument).where(KBDocument.title.ilike(f"%{query}%")).limit(1)
            return db.execute(stmt).scalar_one_or_none()

    def get_by_source_path(self, source_path: str) -> KBDocument | None:
        with SessionLocal() as db:
            stmt = select(KBDocument).where(KBDocument.source_path == source_path)
            return db.execute(stmt).scalar_one_or_none()

    def upsert(self, source_path: str, file_name: str, title: str, metadata_json: dict) -> KBDocument:
        with SessionLocal() as db:
            stmt = select(KBDocument).where(KBDocument.source_path == source_path)
            record = db.execute(stmt).scalar_one_or_none()

            if record is None:
                record = KBDocument(
                    source_path=source_path,
                    file_name=file_name,
                    title=title,
                    metadata_json=metadata_json
                )
                db.add(record)
            else:
                record.file_name = file_name
                record.title = title
                record.metadata_json = metadata_json

            db.commit()
            db.refresh(record)
            return record


kb_document_repo = KBDocumentRepo()
