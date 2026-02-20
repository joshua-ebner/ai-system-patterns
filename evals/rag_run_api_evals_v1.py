"""
Offline evaluation runner for the RAG FastAPI service.

This script sends structured evaluation queries to the running
RAG API (/query endpoint) and logs system-level retrieval results.

Each evaluation case includes:
- a query
- expected document sources
- a must_refuse flag

For each API response, this script records:
- the generated answer
- model refusal behavior
- retrieved sources
- retrieval hit/miss outcome
- latency
- HTTP status

Outputs are appended to:
    evals/rag_api_eval_results_v1.jsonl

Usage:
    1) Start the RAG FastAPI service
    2) Run:
        python evals/rag_run_api_evals_v1.py

This runner provides baseline end-to-end evaluation of:
- retrieval correctness
- refusal logic
- API reliability
- system latency

Future versions may integrate:
- query enhancement
- retrieval validation
- LangSmith trace tagging
- automated miss harvesting
"""

import json
import time
from pathlib import Path

import requests


# -------------------------
# Config
# -------------------------
API_URL = "http://127.0.0.1:8000/query"

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_CASE_FILE = BASE_DIR / "evals/rag_eval_queries_v1.json"
EVAL_LOG_FILE = BASE_DIR / "evals/rag_api_eval_results_v1.jsonl"


TIMEOUT = 60

# LOGGING HELPER FUNCTION
def log_eval_result(payload: dict):
    EVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")

# -------------------------
# Main
# -------------------------
def main():

    with open(EVAL_CASE_FILE, "r") as eval_file:
        eval_data = json.load(eval_file)

    total_queries = len(eval_data)

    retrieval_hits = 0
    correct_refusals = 0
    total_passes = 0
    total_latency = 0.0

    print("\n==== API RAG Eval v1 ====\n")

    for case in eval_data:

        eval_id = case["id"]
        query_text = case["query"]
        expected_sources = set(case["expected_sources"])
        must_refuse = case["must_refuse"]

        print(f"--- {eval_id} ---")
        print(f"Query: {query_text}")

        start_time = time.time()

        response = requests.post(
            API_URL,
            json={"query": query_text},
            timeout=TIMEOUT,
        )

        latency = time.time() - start_time
        total_latency += latency

        response_data = response.json()
        
        answer = response_data["answer"]
        model_refused = response_data["refused"]
        refusal_reason = response_data["refusal_reason"]

        # Build retrieved source set clearly
        retrieved_sources = set()

        source_list = response_data.get("sources", [])
        for source_item in source_list:
            source_name = source_item["source"]
            retrieved_sources.add(source_name)

        print(f"Answer: {answer}")
        print(f"Refused: {model_refused}")
        print(f"Sources: {retrieved_sources}")
        print(f"Latency: {latency:.2f}s")

        # -------------------------
        # Eval logic
        # -------------------------
        if must_refuse:

            if model_refused:
                correct_refusals += 1
                total_passes += 1
                passed = True
                print("✓ Correct refusal\n")
            else:
                passed = False
                print("✗ Should have refused\n")
            log_eval_result({
                "ts": time.time(),
                "eval_id": eval_id,
                "query": query_text,
                "answer": answer,
                "must_refuse": True,
                "refused": model_refused,
                "refusal_reason": refusal_reason,
                "passed": passed,
                "expected_sources": sorted(expected_sources),
                "retrieved_sources": sorted(retrieved_sources),
                "latency_sec": latency,
                "http_status": response.status_code,
            })
            continue

        # Not must_refuse
        if model_refused:
            print("✗ Unexpected refusal\n")
            log_eval_result({
                "ts": time.time(),
                "eval_id": eval_id,
                "query": query_text,
                "answer": answer,
                "must_refuse": False,
                "refused": True,
                "refusal_reason": refusal_reason,
                "passed": False,
                "expected_sources": sorted(expected_sources),
                "retrieved_sources": sorted(retrieved_sources),
                "hit": False,
                "latency_sec": latency,
                "http_status": response.status_code,
            })
            continue

        # Check overlap
        hit = False
        for source in retrieved_sources:
            if source in expected_sources:
                hit = True
                break

        if hit:
            retrieval_hits += 1
            total_passes += 1
            print("✓ Retrieval hit\n")
        else:
            print("✗ Wrong sources\n")
        log_eval_result({
            "ts": time.time(),
            "eval_id": eval_id,
            "query": query_text,
            "answer": answer,
            "must_refuse": False,
            "refused": False,
            "refusal_reason": refusal_reason,
            "passed": hit,
            "expected_sources": sorted(expected_sources),
            "retrieved_sources": sorted(retrieved_sources),
            "hit": hit,
            "latency_sec": latency,
            "http_status": response.status_code,
        })

    # -------------------------
    # Summary
    # -------------------------
    print("\n==== Summary ====")
    print(f"Total queries: {total_queries}")
    print(f"Retrieval hit rate: {retrieval_hits}/{total_queries}")
    print(f"Correct refusals: {correct_refusals}")
    print(f"Overall passes: {total_passes}/{total_queries}")
    print(f"Avg latency: {total_latency/total_queries:.2f}s")
    print("==================\n")


if __name__ == "__main__":
    main()
