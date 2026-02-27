"""
Run comparison utility for agent evaluation summaries.

This script compares two run-level summary JSON files produced by the
offline agent evaluation runner and reports metric deltas.

It is intended for regression detection across agent versions.

Compared metrics:
- pass_rate
- failed count
- avg_latency_sec
- failed_eval_ids (new vs resolved failures)

Typical usage:
    python evals/tools/compare_runs.py \
        --baseline evals/agent/logs/baselines/<baseline_summary>.json \
        --current evals/agent/logs/runs/<current_summary>.json

Optional flags:
    --fail-on-regression
        Exit with non-zero status if a regression is detected
        (pass_rate decreases or failed count increases).

This tool enables:
- Structured regression tracking
- CI integration for automatic failure detection
- Controlled iteration on agent logic
- Reproducible evaluation discipline

It operates only on summary artifacts and does not execute
any evaluation runs directly.
"""


import argparse
import json
import sys
from pathlib import Path


def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two agent eval summary JSON files."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline summary JSON")
    parser.add_argument("--current", required=True, help="Path to current summary JSON")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero status if regression detected",
    )

    args = parser.parse_args()

    baseline = load_summary(Path(args.baseline))
    current = load_summary(Path(args.current))

    # Extract metrics
    base_pass = baseline["pass_rate"]
    curr_pass = current["pass_rate"]

    base_failed = baseline["failed"]
    curr_failed = current["failed"]

    base_latency = baseline["avg_latency_sec"]
    curr_latency = current["avg_latency_sec"]

    base_failed_ids = set(baseline.get("failed_eval_ids", []))
    curr_failed_ids = set(current.get("failed_eval_ids", []))

    # Compute deltas
    delta_pass = curr_pass - base_pass
    delta_failed = curr_failed - base_failed
    delta_latency = curr_latency - base_latency

    new_failures = sorted(curr_failed_ids - base_failed_ids)
    resolved_failures = sorted(base_failed_ids - curr_failed_ids)

    # Print report
    print()
    print(f"Baseline: {baseline['agent_version']} ({base_pass:.2%})")
    print(f"Current:  {current['agent_version']} ({curr_pass:.2%})")
    print()
    print(f"Δ pass_rate: {delta_pass:+.2%}")
    print(f"Δ avg_latency_sec: {delta_latency:+.2f}s")
    print(f"Δ failed: {delta_failed:+d}")
    print()
    print(f"New failures: {new_failures if new_failures else 'none'}")
    print(f"Resolved failures: {resolved_failures if resolved_failures else 'none'}")
    print()

    regression_detected = delta_pass < 0 or delta_failed > 0

    if regression_detected:
        print("Regression detected.")
        if args.fail_on_regression:
            sys.exit(1)
    else:
        print("No regression detected.")

    print()


if __name__ == "__main__":
    main()