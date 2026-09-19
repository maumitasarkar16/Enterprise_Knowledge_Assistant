"""Deterministic overlapping word chunking."""

import hashlib
import re
from src.models import Document, Chunk

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_documents(
    documents: list[Document],
    chunk_size_words: int = 220,
    overlap_words: int = 45,
) -> list[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words")

    chunks = []
    step = chunk_size_words - overlap_words

    for document in documents:
        words = _normalize(document.text).split()
        for chunk_index, start in enumerate(range(0, len(words), step)):
            window = words[start:start + chunk_size_words]
            if not window:
                continue
            text = " ".join(window)
            raw = f"{document.metadata['path']}::{chunk_index}::{text}"
            chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "word_start": start,
                        "word_end": start + len(window) - 1,
                    },
                )
            )
            if start + chunk_size_words >= len(words):
                break
    return chunks
