from app.db.session import engine
from app.models.session_state import Base
from app.models.kb_document import KBDocument
from app.models.kb_chunk import KBChunk


def init_db():
    Base.metadata.create_all(bind=engine)
