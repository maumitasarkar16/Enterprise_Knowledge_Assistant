"""Cross-Encoder second-stage reranking."""

import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str, hf_token: str | None = None):
        self.model = CrossEncoder(
            model_name,
            token=hf_token,
            activation_fn=torch.nn.Sigmoid(),
        )

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(
            candidates, key=lambda x: x.rerank_score, reverse=True
        )[:top_k]
