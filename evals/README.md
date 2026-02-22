# Evals

This directory contains evaluation datasets, runners, and logs for the RAG service and agent layer.

The goal is to make system quality **measurable and repeatable**, not anecdotal.

Evals are intentionally lightweight and JSON-based so they are easy to extend and automate.

---

# Structure

## Query Sets

### `rag_eval_queries_v1.json`

Baseline RAG evaluation set.

Mix of in-scope and out-of-scope queries.

Includes:

- `query`
- `expected_sources`
- `must_refuse`
- notes for human reference

Used to evaluate retrieval quality and refusal behavior.

---

### `agent_eval_queries_v1.json`

Agent-level evaluation set.

Tests agent behavior on top of RAG.

Focuses on:

- Proper refusals  
- Stable answering on in-scope queries

Uses:

- `query`
- `category`
- `must_refuse`

---

## Runners

### `rag_run_api_evals_v1.py`

Runs evals against the RAG FastAPI service.

Measures:

- Retrieval hit rate  
- Refusal correctness  
- Latency  
- Source overlap  

Writes JSONL logs for later analysis.

---

### `agent_run_evals_v1.py`

Runs evals directly against the LangGraph agent.

Measures:

- Correct refusals  
- Unexpected refusals  
- Overall pass rate  
- Latency  

This evaluates orchestration behavior, not raw retrieval.

---

## Logs & Analysis

### `rag_api_eval_results_v1.jsonl`

Raw per-query logs from RAG API evals.

---

### `analyze_rag_api_eval_logs.py`

Utility script for summarizing RAG eval logs.

---

### `eval_log.md`

Human-written notes summarizing past eval runs, thresholds, and observations.

Used as a lightweight experiment journal.

---

# How to Run

## RAG API evals

Start the RAG API:

```bash
uvicorn apps.rag.rag_api:app --reload
```

Run these commands from the project root:

```bash
python -m evals.rag_run_api_evals_v1
```

## Agent evals

Run directly:
```bash
python -m evals.agent_run_evals_v1
```

(No API server required; this invokes the agent graph directly.)

---

# Philosophy

These evals are:

- Small
- Deterministic
- Cheap to run
- Easy to extend

They are not a benchmark suite — they are a regression and sanity-check tool for development.

If you change:
- Chunking
- Embeddings
- Distance thresholds
- Agent logic

You should rerun evals and compare results.