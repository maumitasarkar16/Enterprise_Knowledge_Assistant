from src.retrieval import HybridRetriever

class FakeEmbedding:
    def embed_query(self, query):
        return [1.0, 0.0]

class FakeStore:
    def search(self, vector, top_k):
        return [
            {"id":"hotel","rank":1},
            {"id":"leave","rank":2},
        ][:top_k]

def test_hybrid_rrf_combines_dense_and_bm25(sample_chunks):
    retriever = HybridRetriever(sample_chunks, FakeStore(), FakeEmbedding(), rrf_k=60)
    results = retriever.search(
        "hotel reimbursement SGD 280",
        dense_top_k=2, sparse_top_k=2, hybrid_top_k=3
    )
    assert results
    assert results[0].chunk.id == "hotel"
    assert results[0].hybrid_score > 0
    assert results[0].dense_rank is not None
    assert results[0].sparse_rank is not None
