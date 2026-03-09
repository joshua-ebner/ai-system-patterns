# Evaluations

This directory contains the evaluation infrastructure for the AI Engineering Patterns repo.

The evaluation system is designed to support:

- structured offline testing of agent and RAG behavior
- reproducible run artifacts
- pinned baselines for regression comparison
- lightweight regression testing and CI gating

The current evaluation stack is organized by domain:

- `agent/` → agent-level offline evals
- `rag/` → RAG-specific evals and analysis
- `tools/` → evaluation utilities such as run comparison

---

## Directory Structure

```text
evals/
├── agent/
│   ├── agent_eval_queries_v1.json
│   ├── agent_run_evals_v1.py
│   ├── LOG_SCHEMA.md
│   └── logs/
│       ├── baselines/
│       ├── regression_tests/
│       └── runs/
├── rag/
│   ├── analyze_rag_api_eval_logs.py
│   ├── RAG_eval_log.md
│   ├── rag_eval_queries_v1.json
│   ├── rag_run_api_evals_v1.py
│   ├── rag_run_retrieval_evals_v1.py
│   └── logs/
│       ├── baselines/
│       └── runs/
├── tools/
│   └── compare_eval_runs.py
└── README.md
```

## Overview

There are three main parts of the evaluation system:

- `agent/` — evaluation assets and logs for the LangGraph agent
- `rag/` — evaluation assets and logs for the standalone RAG API
- `tools/` — utility scripts for comparing evaluation results

## Agent Evaluation

The agent evaluation flow tests the end-to-end behavior of the LangGraph agent.

Each evaluation case includes:

- a query
- a `must_refuse` flag
- an optional category

The agent eval runner executes all cases, records per-example results to JSONL, and writes a run summary JSON containing aggregate metrics.

### Main files

- `agent/agent_eval_queries_v1.json` — evaluation dataset
- `agent/agent_run_evals_v1.py` — offline agent evaluation runner
- `agent/LOG_SCHEMA.md` — documentation for the structured log format

### Logged metrics

Per-example logs include:

- query
- answer
- refusal detection
- pass/fail outcome
- latency
- retrieval diagnostics

Run summaries include:

- total
- passed
- failed
- pass_rate
- avg_latency_sec
- failed_eval_ids
- avg_retrieved_count
- avg_top_distance
- avg_retry_count

## Agent Log Directories

The agent evaluation system separates artifacts by purpose.

### `logs/runs/`

This directory contains generated artifacts from actual evaluation runs.

These files are machine-generated outputs and represent the current state of the system under test.

### `logs/baselines/`

This directory contains pinned baseline artifacts used for regression comparison.

A baseline represents a known-good reference run.

### `logs/regression_tests/`

This directory contains intentional regression fixtures used to validate the regression comparison script.

These are not real runs. They are test inputs for the comparison tooling.

## RAG Evaluation

The RAG evaluation flow tests the standalone RAG API directly.

This layer is useful for isolating retrieval and answer behavior outside the agent orchestration layer.

### Main files

- `rag/rag_eval_queries_v1.json` — RAG evaluation dataset
- `rag/rag_run_api_evals_v1.py` — offline RAG API evaluation runner
- `rag/rag_run_retrieval_evals_v1.py` — retrieval-focused evaluation runner
- `rag/analyze_rag_api_eval_logs.py` — analysis utility for RAG API logs
- `rag/RAG_eval_log.md` — notes on earlier RAG evaluation iterations

## Comparison Tools

The `tools/` directory contains scripts that operate on evaluation outputs.

### `tools/compare_eval_runs.py`

This script compares two agent run summary JSON files and reports metric deltas.

It is intended for regression detection and CI enforcement.

It compares summary artifacts, not raw JSONL logs.

Typical use:

- compare a new run summary against a pinned baseline
- fail if pass rate degrades beyond tolerance
- print metric deltas for review

## Evaluation Architecture

The evaluation system is organized around five core pieces:

### Dataset

Evaluation datasets define the test cases for each layer.

Examples:

- `agent/agent_eval_queries_v1.json`
- `rag/rag_eval_queries_v1.json`

These datasets provide the fixed prompts and refusal expectations used during offline testing.

### Run Logs

Each evaluation run produces machine-readable artifacts under `logs/runs/`.

For the agent evals, this includes:

- per-example JSONL logs
- a run-level summary JSON

These artifacts make runs inspectable and reproducible.

### Baselines

Pinned baseline summaries live under `logs/baselines/`.

These represent known-good reference runs and are used to detect regressions over time.

### Regression Detection

Regression detection is handled by:

- `tools/compare_eval_runs.py`

This compares a new run summary against a pinned baseline and checks for regressions in:

- pass rate
- failure count
- latency
- retry behavior

### CI Enforcement

The regression check is enforced automatically in GitHub Actions by:

- `.github/workflows/eval_regression.yml`

The CI workflow:

1. restores or builds the vectorstore
2. starts the RAG API
3. runs the agent eval suite
4. finds the newest summary
5. compares it against the pinned baseline

If a regression is detected, the workflow exits with a failure code.

## Typical Workflow

A normal evaluation workflow looks like this:

1. Run an offline evaluation
2. Inspect the generated summary JSON
3. Compare the new summary against a pinned baseline
4. Detect regressions before merging or deploying changes

## Running Evaluations

Run the agent evaluation:

```bash
python -m evals.agent.agent_run_evals_v1
```

Run the RAG API evaluation:

```bash
python -m evals.rag.rag_run_api_evals_v1
```

Run the retrieval-focused RAG evaluation:

```bash
python -m evals.rag.rag_run_retrieval_evals_v1
```

Compare a new agent run against a baseline:

```bash
python evals/tools/compare_eval_runs.py \
  --current evals/agent/logs/runs/<current_summary>.json \
  --baseline evals/agent/logs/baselines/<baseline_summary>.json
```

## Design Principles

The evaluation system is built around a few principles:

- structured outputs over ad hoc inspection
- reproducible summaries over vague impressions
- pinned baselines over informal memory
- regression detection over silent degradation

This makes the repo more representative of real AI system engineering, where evaluation is part of the product lifecycle.