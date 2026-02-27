"""
Offline evaluation runner for the LangGraph RAG agent.

This script executes a set of structured evaluation queries against the
agent and logs the results to a JSONL file for analysis.

Each evaluation case includes:
- a query
- a must_refuse flag (whether the agent should decline to answer)

For each run, this script records:
- the agent's response
- refusal detection (string heuristic, v1)
- pass/fail outcome
- latency

Outputs are recorded in:
    evals/agent/logs/runs/<timestamp>_agent_eval_results_v2_escalation.jsonl

Usage:
    python evals/agent/agent_run_evals_v1.py

This runner serves as the baseline offline eval loop for:
- refusal behavior
- reliability tracking
- latency measurement

Future versions may integrate:
- structured refusal signals
- LangSmith trace tagging
- automated miss harvesting
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time

from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from apps.agent.agent_graph import agent

import subprocess


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_CASE_FILE = BASE_DIR / "evals/agent/agent_eval_queries_v1.json"


# -------------------------
# Run metadata (schema v2)
# -------------------------
EVAL_SCHEMA_VERSION = 2
AGENT_VERSION = "v2_escalation"   # <-- update when you change agent logic
RAG_VERSION = "v1"               # <-- update when retrieval pipeline changes
MODEL_NAME = "gpt-4o-mini"       # <-- must match the model used in the agent graph
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_ID = f"{RUN_TIMESTAMP}_{AGENT_VERSION}"

# OUTPUT FILE
EVAL_LOG_FILE = BASE_DIR / f"evals/agent/logs/runs/{RUN_TIMESTAMP}_agent_eval_results_{AGENT_VERSION}.jsonl"



def get_git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

GIT_COMMIT = get_git_commit_short()


# -------------------------
# Logging helper
# -------------------------
def log_eval_result(payload: dict):
    """Append one evaluation result to the JSONL log."""
    EVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")



#---------------------------
# RUN PERFORMANCE SUMMARIZER
#---------------------------
def write_run_summary(
    run_id: str,
    agent_version: str,
    rag_version: str,
    model: str,
    git_commit: str,
    total: int,
    passed: int,
    correct_refusals: int,
    unexpected_refusals: int,
    avg_latency_sec: float,
    failed_eval_ids: list[str],
):
    """
    Write run-level summary metrics for a single agent eval run.
    """

    os.makedirs("evals/agent/logs/runs", exist_ok=True)

    summary = {
        "run_id": run_id,
        "agent_version": agent_version,
        "rag_version": rag_version,
        "model": model,
        "git_commit": git_commit,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "correct_refusals": correct_refusals,
        "unexpected_refusals": unexpected_refusals,
        "pass_rate": passed / total if total else 0.0,
        "avg_latency_sec": avg_latency_sec,
        "failed_eval_ids": failed_eval_ids,
    }

    output_path = f"evals/agent/logs/runs/{run_id}_summary.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {output_path}")



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

    results = []

    print(f"\n==== Agent Eval (schema v{EVAL_SCHEMA_VERSION}) ====\n")

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

        
        # initialize RAG retrieval variables
        retrieved_count = None
        top_distance = None
        retry_count = None

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)],
                 "rag_result": None,
                 "retry_count": 0,
                 }
            )
            answer = result["messages"][-1].content
            rag_result = result.get("rag_result") or {}
            retrieved_count = rag_result.get("retrieved_count")
            top_distance = rag_result.get("top_distance")
            retry_count = result.get("retry_count")
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

        results.append({
            "eval_id": eval_id,
            "passed": passed,
            })

        # -------------------------
        # Log structured result
        # -------------------------
        log_eval_result({
            # --- Run-level metadata (NEW) ---
            "eval_schema_version": EVAL_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "agent_version": AGENT_VERSION,
            "rag_version": RAG_VERSION,
            "model": MODEL_NAME,
            "git_commit": GIT_COMMIT,
        
            # --- Per-example data (existing) ---
            "ts": time.time(),
            "eval_id": eval_id,
            "query": query,
            "category": category,
            "answer": answer,
            "must_refuse": must_refuse,
            "refused": refused,
            "passed": passed,
            "latency_sec": latency,
            "log_retrieved_count": retrieved_count,
            "log_top_distance": top_distance,
            "log_retry_count": retry_count,
        })
        #

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

    failed_eval_ids = [r["eval_id"] for r in results if not r["passed"]]
    
    write_run_summary(
        run_id=RUN_ID,
        agent_version=AGENT_VERSION,
        rag_version=RAG_VERSION,
        model=MODEL_NAME,
        git_commit=GIT_COMMIT,
        total=total,
        passed=total_passes,
        correct_refusals=correct_refusals,
        unexpected_refusals=unexpected_refusals,
        avg_latency_sec=(total_latency / total) if total else 0.0,
        failed_eval_ids=failed_eval_ids,
    )


if __name__ == "__main__":
    main()
