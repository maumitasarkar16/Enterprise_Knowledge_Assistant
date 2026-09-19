"""Dense + BM25 hybrid retrieval using Reciprocal Rank Fusion."""

import re
from rank_bm25 import BM25Okapi
from src.models import Chunk, RetrievedChunk

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

class HybridRetriever:
    def __init__(self, chunks, vector_store, embedding_model, rrf_k=60):
        self.chunks = chunks
        self.chunk_by_id = {c.id: c for c in chunks}
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def dense_search(self, query: str, top_k: int) -> list[str]:
        vector = self.embedding_model.embed_query(query)
        return [r["id"] for r in self.vector_store.search(vector, top_k)]

    def sparse_search(self, query: str, top_k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        indexes = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:min(top_k, len(scores))]
        return [self.chunks[i].id for i in indexes]

    def search(self, query, dense_top_k, sparse_top_k, hybrid_top_k):
        dense_ids = self.dense_search(query, dense_top_k)
        sparse_ids = self.sparse_search(query, sparse_top_k)
        fused = {}

        for rank, cid in enumerate(dense_ids, 1):
            item = fused.setdefault(cid, RetrievedChunk(self.chunk_by_id[cid]))
            item.dense_rank = rank
            item.hybrid_score += 1.0 / (self.rrf_k + rank)

        for rank, cid in enumerate(sparse_ids, 1):
            item = fused.setdefault(cid, RetrievedChunk(self.chunk_by_id[cid]))
            item.sparse_rank = rank
            item.hybrid_score += 1.0 / (self.rrf_k + rank)

        return sorted(
            fused.values(), key=lambda x: x.hybrid_score, reverse=True
        )[:hybrid_top_k]
