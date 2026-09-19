"""Persistent local Chroma vector database."""

from pathlib import Path
import chromadb
from src.models import Chunk

class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        name = self.collection.name
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        for start in range(0, len(chunks), 100):
            batch = chunks[start:start + 100]
            emb = embeddings[start:start + 100]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=emb,
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for rank, (cid, text, meta, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ),
            1,
        ):
            rows.append({
                "id": cid, "text": text, "metadata": meta,
                "distance": float(distance), "rank": rank
            })
        return rows
