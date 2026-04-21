from sentence_transformers import SentenceTransformer
from app.core.config import settings


class LocalEmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        return embedding.tolist()


local_embedding_service = LocalEmbeddingService()
