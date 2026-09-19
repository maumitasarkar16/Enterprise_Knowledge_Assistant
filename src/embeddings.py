"""Local Sentence-Transformers embeddings."""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LocalEmbeddingModel:
    def __init__(self, model_name: str, hf_token: str | None = None):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, token=hf_token)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text], show_progress_bar=False, normalize_embeddings=True
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()
