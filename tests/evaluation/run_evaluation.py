"""Live end-to-end evaluation for routing, RAG, not-found and memory."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.pipeline import RAGPipeline
from src.llm import NOT_FOUND_MESSAGE

HERE = Path(__file__).parent
def load_json(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))

def mark(ok, case_id, text):
    print(f"{case_id:<12} {'PASS' if ok else 'FAIL'}  {text}")
    return int(ok), int(not ok)

def main():
    print("=" * 76)
    print("ENTERPRISE KNOWLEDGE ASSISTANT - LIVE DUAL-MODE EVALUATION")
    print("=" * 76)
    pipeline = RAGPipeline(get_settings())
    pipeline.initialize()
    passed = failed = 0

    print("\nROUTER")
    for case in load_json("router_test_cases.json"):
        mode = pipeline.llm.route_query(case["question"], [], 8)
        p, f = mark(mode == case["expected_mode"], case["id"], f"{case['question']} -> {mode}")
        passed += p; failed += f

    print("\nGENERAL")
    result = pipeline.answer("What is the capital of India?", [])
    ok = result.mode == "GENERAL" and "new delhi" in result.answer.lower() and not result.sources
    p, f = mark(ok, "GEN-001", result.answer); passed += p; failed += f

    print("\nENTERPRISE RAG")
    for case in load_json("rag_test_cases.json"):
        result = pipeline.answer(case["question"], [])
        text = result.answer.lower()
        keywords_ok = all(k.lower() in text for k in case["expected_keywords"])
        source_ok = any(s["source"] == case["expected_source"] for s in result.sources)
        ok = result.mode == "ENTERPRISE" and keywords_ok and source_ok
        p, f = mark(ok, case["id"], case["question"]); passed += p; failed += f

    print("\nENTERPRISE NOT-FOUND")
    for case in load_json("hallucination_test_cases.json"):
        result = pipeline.answer(case["question"], [])
        ok = (
            result.mode == "ENTERPRISE"
            and result.answer.strip() == NOT_FOUND_MESSAGE
            and not result.sources
        )
        p, f = mark(ok, case["id"], case["question"]); passed += p; failed += f

    print("\nMEMORY")
    history = []
    q1 = "What is the hotel reimbursement cap?"
    r1 = pipeline.answer(q1, history)
    history += [{"role":"user","content":q1},{"role":"assistant","content":r1.answer}]
    q2 = "When can it be exceeded?"
    r2 = pipeline.answer(q2, history)
    ok = (
        r2.mode == "ENTERPRISE"
        and "hotel" in r2.retrieval_query.lower()
        and "reimbursement" in r2.retrieval_query.lower()
        and any(s["source"] == "travel_policy.md" for s in r2.sources)
    )
    p, f = mark(ok, "MEM-LIVE-1", f"{q2} -> {r2.retrieval_query}")
    passed += p; failed += f

    print("\n" + "=" * 76)
    print(f"PASSED: {passed} | FAILED: {failed}")
    print("=" * 76)
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
