"""
Migration utility for upgrading legacy agent eval JSONL logs
to the schema v2 format.

This script reads existing JSONL evaluation logs that lack
run-level metadata and writes new *_migrated.jsonl files with:

- eval_schema_version
- run_id
- agent_version
- rag_version
- model
- git_commit

Design principles:
- Non-destructive (original files are not modified)
- Idempotent (does not overwrite fields that already exist)
- Explicit per-file metadata configuration

This allows historical eval runs to be brought into alignment
with the current structured evaluation schema without rewriting
repository history.
"""


import json
from pathlib import Path


EVAL_SCHEMA_VERSION = 2


def migrate_jsonl(
    src_path: Path,
    dst_path: Path,
    *,
    run_id: str,
    agent_version: str,
    rag_version: str,
    model: str,
    git_commit: str = "unknown",
):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            obj.setdefault("eval_schema_version", EVAL_SCHEMA_VERSION)
            obj.setdefault("run_id", run_id)
            obj.setdefault("agent_version", agent_version)
            obj.setdefault("rag_version", rag_version)
            obj.setdefault("model", model)
            obj.setdefault("git_commit", git_commit)

            f_out.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]

    migrations = [
        {
            "src": base / "evals/agent/2026-02-17_agent_eval_results_v1_baseline.jsonl",
            "dst": base / "evals/agent/2026-02-17_agent_eval_results_v1_baseline_migrated.jsonl",
            "run_id": "2026-02-17_v1_baseline",
            "agent_version": "v1_baseline",
            "rag_version": "v1",
            "model": "gpt-4o-mini",
            "git_commit": "unknown",
        },
    ]

    for m in migrations:
        migrate_jsonl(
            m["src"],
            m["dst"],
            run_id=m["run_id"],
            agent_version=m["agent_version"],
            rag_version=m["rag_version"],
            model=m["model"],
            git_commit=m.get("git_commit", "unknown"),
        )

    print("Migration complete.")