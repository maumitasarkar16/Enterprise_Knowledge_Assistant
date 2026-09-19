from src.models import RetrievedChunk
from src.reranker import CrossEncoderReranker

class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return [0.15, 0.40, 0.92]

def test_reranker_changes_candidate_order(sample_chunks):
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = FakeCrossEncoder()
    candidates = [RetrievedChunk(c) for c in sample_chunks]
    ranked = reranker.rerank("Who approves production access?", candidates, 3)
    assert ranked[0].chunk.id == "security"
    assert ranked[0].rerank_score == 0.92
