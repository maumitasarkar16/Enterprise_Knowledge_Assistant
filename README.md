# Enterprise Knowledge Assistant with Advanced RAG
**Developer: Maumita Sarkar**

A locally runnable dual-mode assistant for the final assessment.

## Dual-mode design

### GENERAL mode
Handles greetings, casual conversation and public/general knowledge using the LLM directly.

Examples:
- `Hello, I am Maumita`
- `What is the capital of India?`
- `Explain Python decorators`

### ENTERPRISE mode
Handles organisation-specific/internal questions using the local Advanced RAG pipeline only.

Examples:
- `What is our hotel reimbursement cap?`
- `How many annual leave days do employees receive?`
- `Who approves privileged production access?`
- `What is our maternity leave policy?` → if absent, returns explicit enterprise not-found.

**Important safety/grounding rule:** an ENTERPRISE question never falls back to general LLM knowledge when retrieval fails.

## Enterprise RAG pipeline
Local documents → chunking → SentenceTransformer embeddings → ChromaDB dense retrieval + BM25 → Reciprocal Rank Fusion → Cross-Encoder reranking → relevance guard → grounded OpenAI answer with `[S#]` citations.

## Conversation memory
Recent chat history is used by the router and by the enterprise standalone-query rewriter. This allows:
1. `What is the hotel reimbursement cap?`
2. `When can it be exceeded?`
3. `Does my manager need to approve that?`

The second and third turns remain ENTERPRISE and are resolved using prior context.

## Run
See `RUN_GUIDE.md`.

```bash
python -m venv .venv
# activate it
pip install -r requirements.txt
# copy .env.example to .env and add OPENAI_API_KEY
streamlit run app.py
```

## Tests
```bash
pytest -v
pytest --cov=src --cov-report=term-missing -v
python tests/evaluation/run_evaluation.py
```

The live evaluation covers router behaviour, GENERAL knowledge, ENTERPRISE RAG, enterprise not-found and multi-turn memory.

## Submission hygiene
Do not submit `.env`, `.venv`, `venv`, Chroma runtime data or real API keys.
