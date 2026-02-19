from dotenv import load_dotenv
load_dotenv()

import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from apps.agent_graph import agent


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

EVAL_CASE_FILE = BASE_DIR / "evals/agent_eval_queries_v1.json"
EVAL_LOG_FILE = BASE_DIR / "evals/agent_eval_results_v1.jsonl"


# -------------------------
# Logging helper
# -------------------------
def log_eval_result(payload: dict):
    """Append one evaluation result to the JSONL log."""
    EVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


# -------------------------
# Main
# -------------------------
def main():

    # -------------------------
    # Load evaluation cases
    # -------------------------
    with open(EVAL_CASE_FILE, "r") as f:
        eval_data = json.load(f)

    total = len(eval_data)

    # -------------------------
    # Metric counters
    # -------------------------
    correct_refusals = 0
    unexpected_refusals = 0
    total_passes = 0
    total_latency = 0.0

    print("\n==== Agent Eval v1 ====\n")

    # -------------------------
    # Main evaluation loop
    # -------------------------
    for case in eval_data:

        eval_id = case["id"]
        query = case["query"]
        must_refuse = case["must_refuse"]
        category = case.get("category", "unknown")

        print(f"--- {eval_id} ({category}) ---")
        print(f"Query: {query}")

        # ---- Invoke agent and time execution ----
        start = time.time()

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)],
                 "rag_result": None,
                 "retry_count": 0,
                 }
            )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"Agent error: {str(e)}"

        latency = time.time() - start
        total_latency += latency

        print(f"Answer: {answer}")
        print(f"Latency: {latency:.2f}s")

        # -------------------------
        # Refusal detection heuristic
        # (string-based for v1 simplicity)
        # -------------------------
        lower = answer.lower()

        refused = (
            "don't have enough relevant context" in lower
            or "not enough relevant context" in lower
        )

        # -------------------------
        # Evaluation logic
        # -------------------------
        if must_refuse:

            if refused:
                correct_refusals += 1
                total_passes += 1
                passed = True
                print("✓ Correct refusal\n")
            else:
                passed = False
                print("✗ Should have refused\n")

        else:

            if refused:
                unexpected_refusals += 1
                passed = False
                print("✗ Unexpected refusal\n")
            else:
                total_passes += 1
                passed = True
                print("✓ Pass\n")

        # -------------------------
        # Log structured result
        # -------------------------
        log_eval_result({
            "ts": time.time(),
            "eval_id": eval_id,
            "query": query,
            "category": category,
            "answer": answer,
            "must_refuse": must_refuse,
            "refused": refused,
            "passed": passed,
            "latency_sec": latency,
        })

    # -------------------------
    # Summary report
    # -------------------------
    print("\n==== Summary ====")
    print(f"Total queries: {total}")
    print(f"Correct refusals: {correct_refusals}")
    print(f"Unexpected refusals: {unexpected_refusals}")
    print(f"Overall passes: {total_passes}/{total}")
    print(f"Avg latency: {total_latency/total:.2f}s")
    print("==================\n")


if __name__ == "__main__":
    main()
