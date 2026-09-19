from src.pipeline import RAGPipeline
from src.llm import NOT_FOUND_MESSAGE
from src.models import RetrievedChunk

class FakeLLM:
    def __init__(self, route="ENTERPRISE"):
        self.route = route
    def route_query(self, question, history, max_messages):
        return self.route
    def answer_general(self, question, history, max_messages):
        return "New Delhi." if "capital of India" in question else "Hello!"
    def rewrite_query(self, question, history, max_messages):
        if history and question == "When can it be exceeded?":
            return "When can the hotel reimbursement cap be exceeded?"
        return question
    def answer_enterprise(self, question, context):
        return "The cap is SGD 280 per night before taxes. [S1]"

class FakeRetriever:
    def __init__(self, chunk): self.chunk = chunk
    def search(self, *args, **kwargs):
        return [RetrievedChunk(self.chunk)]

class FakeReranker:
    def __init__(self, score): self.score = score
    def rerank(self, query, candidates, top_k):
        for c in candidates: c.rerank_score = self.score
        return candidates[:top_k]

def make_pipeline(settings, chunk, score=0.9, route="ENTERPRISE"):
    p = RAGPipeline(settings)
    p.llm = FakeLLM(route)
    p.retriever = FakeRetriever(chunk)
    p.reranker = FakeReranker(score)
    return p

def test_general_question_skips_rag(settings, sample_chunks):
    p = make_pipeline(settings, sample_chunks[1], route="GENERAL")
    result = p.answer("What is the capital of India?", [])
    assert result.mode == "GENERAL"
    assert "New Delhi" in result.answer
    assert result.sources == []
    assert result.retrieval_query == ""

def test_enterprise_grounded_answer_contains_source(settings, sample_chunks):
    p = make_pipeline(settings, sample_chunks[1], 0.9, "ENTERPRISE")
    result = p.answer("What is our hotel cap?", [])
    assert result.mode == "ENTERPRISE"
    assert "SGD 280" in result.answer
    assert result.sources[0]["source"] == "travel_policy.md"

def test_enterprise_low_relevance_returns_not_found(settings, sample_chunks):
    p = make_pipeline(settings, sample_chunks[1], 0.01, "ENTERPRISE")
    result = p.answer("Who is our CEO?", [])
    assert result.mode == "ENTERPRISE"
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
