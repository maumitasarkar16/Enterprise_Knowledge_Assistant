# Test Plan

## Goal
Provide functional coverage across the Advanced RAG lifecycle rather than claiming that an LLM is mathematically hallucination-free.

## Coverage areas
- document loading and invalid inputs
- chunking and overlap
- deterministic IDs and metadata
- lexical tokenization
- dense/vector-store path
- BM25 sparse retrieval
- RRF hybrid fusion
- Cross-Encoder reranking
- relevance gating
- source construction/citations
- not-found handling
- conversation memory/query rewriting
- memory reset in Streamlit
- live end-to-end evaluation

## Test layers
1. `tests/unit/`: deterministic component tests.
2. `tests/integration/`: component interaction using fakes/mocks where practical.
3. `tests/memory/`: explicit proof that prior conversation changes the standalone retrieval query.
4. `tests/evaluation/`: machine-readable expected input/output datasets and live evaluation runner.
5. Manual Streamlit acceptance testing for UX and session reset.
