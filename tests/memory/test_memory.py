"""Tests proving enterprise follow-ups use conversation history."""

from src.pipeline import RAGPipeline
from src.models import RetrievedChunk

class MemoryAwareLLM:
    def route_query(self, question, history, max_messages):
        history_text = " ".join(m["content"] for m in history).lower()
        if "hotel" in history_text or "learning budget" in history_text:
            return "ENTERPRISE"
        return "ENTERPRISE"
    def rewrite_query(self, question, history, max_messages):
        history_text = " ".join(m["content"] for m in history).lower()
        if question == "When can it be exceeded?" and "hotel" in history_text:
            return "When can the hotel reimbursement cap be exceeded?"
        if question == "Who qualifies for it?" and "learning budget" in history_text:
            return "Who qualifies for the learning budget?"
        return question
    def answer_enterprise(self, question, context):
        if "hotel" in context.lower():
            return "A higher hotel rate may be reimbursed with manager approval. [S1]"
        return "Permanent employees after six months qualify. [S1]"

class CapturingRetriever:
    def __init__(self, chunk):
        self.chunk = chunk
        self.last_query = None
    def search(self, query, *args):
        self.last_query = query
        return [RetrievedChunk(self.chunk)]

class PassReranker:
    def rerank(self, query, candidates, top_k):
        for c in candidates: c.rerank_score = 0.95
        return candidates[:top_k]

def build(settings, chunk):
    p = RAGPipeline(settings)
    p.llm = MemoryAwareLLM()
    p.retriever = CapturingRetriever(chunk)
    p.reranker = PassReranker()
    return p

def test_hotel_followup_uses_history(settings, sample_chunks):
    p = build(settings, sample_chunks[1])
    history = [
        {"role":"user","content":"What is the hotel reimbursement cap?"},
        {"role":"assistant","content":"The cap is SGD 280 per night. [S1]"},
    ]
    result = p.answer("When can it be exceeded?", history)
    assert result.mode == "ENTERPRISE"
    assert "hotel reimbursement cap" in result.retrieval_query.lower()
    assert result.sources[0]["source"] == "travel_policy.md"

def test_same_followup_without_history_is_not_rewritten(settings, sample_chunks):
    p = build(settings, sample_chunks[1])
    result = p.answer("When can it be exceeded?", [])
    assert result.retrieval_query == "When can it be exceeded?"

def test_learning_budget_pronoun_resolution(settings, sample_chunks):
    p = build(settings, sample_chunks[0])
    history = [
        {"role":"user","content":"What is the learning budget?"},
        {"role":"assistant","content":"It is SGD 1,200 per year."},
    ]
    result = p.answer("Who qualifies for it?", history)
    assert result.retrieval_query == "Who qualifies for the learning budget?"
