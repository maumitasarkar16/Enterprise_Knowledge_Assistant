"""Shared pytest fixtures using lightweight fakes where possible."""

from types import SimpleNamespace
import pytest
from src.models import Chunk, RetrievedChunk

@pytest.fixture
def settings():
    return SimpleNamespace(
        dense_top_k=12, sparse_top_k=12, hybrid_top_k=15,
        rerank_top_k=5, min_rerank_score=0.12, max_history_messages=8,
        chunk_size_words=220, chunk_overlap_words=45, rrf_k=60,
    )

@pytest.fixture
def sample_chunks():
    return [
        Chunk("leave", "Full-time employees receive 20 days annual leave.",
              {"source":"employee_handbook.md","path":"employee_handbook.md",
               "extension":".md","chunk_index":0}),
        Chunk("hotel", "The standard hotel reimbursement cap is SGD 280 per night before taxes.",
              {"source":"travel_policy.md","path":"travel_policy.md",
               "extension":".md","chunk_index":0}),
        Chunk("security", "Privileged production access requires engineering manager and system owner approval.",
              {"source":"it_security_policy.md","path":"it_security_policy.md",
               "extension":".md","chunk_index":0}),
    ]
