"""Environment-driven application configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    huggingface_api_key: str | None
    gemini_api_key: str | None

    openai_chat_model: str = "gpt-5-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    data_dir: str = "data/sample_docs"
    chroma_dir: str = "data/chroma_db"
    collection_name: str = "enterprise_knowledge"

    chunk_size_words: int = 220
    chunk_overlap_words: int = 45

    dense_top_k: int = 12
    sparse_top_k: int = 12
    hybrid_top_k: int = 15
    rerank_top_k: int = 5
    rrf_k: int = 60

    # Tune on a labelled evaluation set for a real enterprise domain.
    min_rerank_score: float = 0.12
    max_history_messages: int = 8

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY", "").strip() or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
        ),
        data_dir=os.getenv("DATA_DIR", "data/sample_docs"),
        chroma_dir=os.getenv("CHROMA_DIR", "data/chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "enterprise_knowledge"),
        min_rerank_score=float(os.getenv("MIN_RERANK_SCORE", "0.12")),
    )
