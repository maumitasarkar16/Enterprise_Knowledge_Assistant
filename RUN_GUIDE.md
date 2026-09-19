# Run Guide

## Prerequisites
- Python 3.11 recommended
- Internet on first run for local Hugging Face model downloads
- OpenAI API key

## Windows
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

## macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add the real OpenAI key. Never commit `.env`.

## Run UI
```bash
streamlit run app.py
```

## Quick functional checks
GENERAL:
- `Hello, I am Maumita` → friendly response; Mode GENERAL
- `What is the capital of India?` → New Delhi; Mode GENERAL

ENTERPRISE:
- `What is our hotel reimbursement cap?` → SGD 280 + source
- `What is our maternity leave policy?` → enterprise not-found

MEMORY:
1. `What is the hotel reimbursement cap?`
2. `When can it be exceeded?`
3. `Does my manager need to approve that?`

Open Diagnostics after turn 2. It should show ENTERPRISE and a rewritten query containing hotel/reimbursement context.

## Automated tests
```bash
pytest -v
```

Coverage:
```bash
pytest --cov=src --cov-report=term-missing -v
```

Only router tests:
```bash
pytest tests/unit/test_router_behavior.py -v
```

Only memory tests:
```bash
pytest tests/memory/test_memory.py -v
```

## Live end-to-end evaluation
```bash
python tests/evaluation/run_evaluation.py
```

This requires the real OpenAI key and tests:
- GENERAL vs ENTERPRISE routing
- general knowledge
- enterprise RAG
- enterprise not-found
- live multi-turn memory
