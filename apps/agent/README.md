# Agent Layer (LangGraph)

This module implements a lightweight LangGraph agent that orchestrates
a RAG system exposed as a FastAPI service.

The agent does **not** perform retrieval itself.
Instead, it treats the RAG API as an external tool and focuses on
orchestration and decision flow.

This demonstrates a production-style separation between:

- Retrieval service (RAG API)
- Agent orchestration layer

---

## Design

The agent:

- Uses LangGraph
- Calls the RAG API via an HTTP tool
- Respects refusal signals from the RAG system
- Logs structured runs for analysis

v1 intentionally keeps logic simple:

- Single-turn interactions
- No memory
- No complex routing
- Thin orchestration wrapper

This keeps the architecture explicit and easy to reason about.

---

## Evaluation & Observability

This agent layer is evaluated using a small set of structured evaluation queries 
(in-scope, borderline, and out-of-scope queries) to verify:

- Proper tool usage
- Correct refusal behavior
- Consistent latency

LangSmith tracing is enabled for step-level observability of
LLM calls and tool execution.

---

## Why This Layer Exists

In real systems, retrieval pipelines and agent logic are often deployed
as separate services. This module mirrors that pattern by treating RAG
as a dedicated microservice and keeping the agent focused on orchestration.

---

## Files

- `rag_query_tool.py`  
  HTTP tool wrapper for the RAG API.

- `agent_graph.py`  
  Minimal LangGraph definition.

- `run_agent.py`  
  CLI runner with structured logging.

---

## Running

From the project root:

Start the RAG API first:

```bash
uvicorn apps.rag.rag_api:app --reload
```

Then run the agent:

```bash
python -m apps.agent.run_agent
```