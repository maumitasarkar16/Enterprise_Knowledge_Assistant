"""Domain data classes shared across RAG components."""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    text: str
    metadata: dict[str, Any]

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]

@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    hybrid_score: float = 0.0
    rerank_score: float = 0.0

@dataclass
class AnswerResult:
    answer: str
    mode: str
    retrieval_query: str = ""
    best_rerank_score: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
