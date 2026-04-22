from sentence_transformers import SentenceTransformer
from app.core.config import settings


class LocalEmbeddingService:
    def __init__(self):
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(settings.embedding_model_name)
        return self.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._get_model().encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        return embedding.tolist()


local_embedding_service = LocalEmbeddingService()
