from src.config import Settings

def test_safe_defaults():
    s = Settings(openai_api_key="x", huggingface_api_key=None, gemini_api_key=None)
    assert s.dense_top_k > 0
    assert s.sparse_top_k > 0
    assert s.rerank_top_k > 0
    assert 0 <= s.min_rerank_score <= 1
