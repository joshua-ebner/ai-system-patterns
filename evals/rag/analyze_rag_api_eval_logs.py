import json
from pathlib import Path
from statistics import mean, median

EVAL_LOG_FILE = Path("evals/rag/logs/runs/rag_api_eval_results_v1.jsonl")


def main():
    rows = []

    with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print("No eval data found.")
        return

    total = len(rows)

    passes = sum(r.get("passed", False) for r in rows)

    must_refuse_cases = [r for r in rows if r.get("must_refuse")]
    correct_refusals = sum(
        r.get("must_refuse") and r.get("refused") for r in rows
    )

    non_refusal_cases = [r for r in rows if not r.get("must_refuse")]
    retrieval_hits = sum(
        r.get("hit", False) for r in non_refusal_cases
    )

    latencies = [r["latency_sec"] for r in rows if "latency_sec" in r]

    unexpected_refusals = [
        r for r in rows
        if (not r.get("must_refuse")) and r.get("refused")
    ]

    failures = [r for r in rows if not r.get("passed")]

    print("\n==== RAG Eval Analysis ====\n")

    print(f"Total queries: {total}")
    print(f"Overall pass rate: {passes}/{total} ({passes/total:.1%})")

    if non_refusal_cases:
        print(
            f"Retrieval hit rate (non-refusal): "
            f"{retrieval_hits}/{len(non_refusal_cases)} "
            f"({retrieval_hits/len(non_refusal_cases):.1%})"
        )

    if must_refuse_cases:
        print(
            f"Correct refusal rate: "
            f"{correct_refusals}/{len(must_refuse_cases)} "
            f"({correct_refusals/len(must_refuse_cases):.1%})"
        )

    if latencies:
        print(f"Avg latency: {mean(latencies):.2f}s")
        print(f"Median latency: {median(latencies):.2f}s")

    print(f"Unexpected refusals: {len(unexpected_refusals)}")

    if failures:
        print("\n--- Failed Queries ---")
        for r in failures[:5]:  # cap at 5
            print(f"- {r['eval_id']}: {r['query']}")

    print("\n========================\n")


if __name__ == "__main__":
    main()
