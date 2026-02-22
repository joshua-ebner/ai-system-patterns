# AI Engineering Patterns

Production-oriented patterns for building reliable RAG and agent systems – with working code, evaluation, and a live API.

This repository focuses on *how real AI systems are engineered*, not just how to call an LLM API.

The goal is to demonstrate practical system design decisions around retrieval, control flow, evaluation, and failure handling.

---

## What This Repository Demonstrates

This repo contains working examples and patterns for:

- Embeddings and vector-based retrieval  
- Token-aware chunking strategies  
- Deterministic ingestion pipelines  
- Agent routing and control flow  
- Prompt-driven decision logic  
- Tool selection and invocation  
- Confidence-based refusal behavior  
- Evaluation of retrieval quality and agent behavior  
- Testing with known inputs and expected outputs  
- Observing failure modes and edge cases  

The emphasis is on correctness, reasoning, and system design — not toy demos or UI wrappers.

---

## Project Goal

The system being built here is an **AI Engineering Assistant**:

A RAG/agent system grounded in real AI engineering documentation (LangChain, OpenAI, and related tooling) that can:

- Retrieve relevant technical knowledge  
- Answer grounded engineering questions  
- Refuse when context is insufficient  
- Support evaluation and testing of behavior  

At the same time, the repository serves as **external proof of RAG and agent engineering ability.**

---
## Architecture (V1)

flowchart LR
  U[User] --> A[Agent Layer\nLangGraph\napps/agent]
  A -->|HTTP POST /query| R[RAG Service\nFastAPI\napps/rag/rag_api.py]

  R --> V[(Chroma\nVector Store)]
  R --> M[(LLM\nChatOpenAI)]

  A --> L[(JSONL Logs\nlogs/agent_runs_v1.jsonl)]
  R --> L2[(JSONL Logs\nlogs/rag_queries_v1.jsonl)]

  A --> S[LangSmith Tracing]
  R --> S

---

## Current Status

This project is being built in phases.

### Phase 1 — RAG Foundations (In Progress)

Focus:

- Documentation ingestion  
- Token-aware chunking  
- Embedding + retrieval pipelines  
- Confidence-based refusal logic  
- Small evaluation set for retrieval quality  

Corpus sources:

- LangChain documentation  
- OpenAI documentation (embeddings and tool use)

Out of scope for Phase 1:

- Agents  
- LangGraph  
- LangSmith  
- Deployment infrastructure  

---
## Data Pipeline Overview

This repository uses a reproducible documentation pipeline
to build an AI Engineering knowledge base.

Raw documentation is not committed to the repo.
Instead, source documents are pulled programmatically.

To fetch the LangChain documentation corpus:

```bash
bash scripts/pull_langchain_docs.sh
```

This will populate:

```text
data/raw/langchain/
```

Subsequent ingestion and chunking steps operate on this data.

---

## RAG API v1

A simple Retrieval-Augmented Generation (RAG) API built with:

- FastAPI
- LangChain
- OpenAI embeddings
- Chroma vector store

This service retrieves relevant document chunks and generates grounded answers with source citations.

---

### Features

- Semantic retrieval over a local vector database
- Distance-threshold filtering
- Grounded answer generation
- Source attribution with snippets
- Refusal for low-relevance queries
- JSONL logging for analysis

---

### Running the API

Start the server:

```bash
uvicorn apps.rag.rag_api:app --reload
```

Open interactive docs:

`http://127.0.0.1:8000/docs`

Health check:
```bash
curl http://127.0.0.1:8000/health`
```

Query:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a retriever in LangChain?"}'
```

Response includes:
- answer
- refused flag (for low-relevance / out-of-scope queries)
- sources with distance + snippet


---

## Evaluation Results

These are ongoing evaluation results for the system.

### v1 Evaluation (RAG API)

I created a structured evaluation set of 16 queries (including in-scope technical questions and deliberate out-of-scope queries) and built an automated runner that tests the full API end-to-end.

**Current v1 Performance:**

- Overall pass rate: **13/16 (81.2%)**
- Retrieval hit rate (answerable queries): **10/13 (76.9%)**
- Correct refusal rate: **3/3 (100%)**
- Average latency: **~2.5 seconds**
- Median latency: **~2.4 seconds**

These metrics provide a baseline for iteration. Future improvements (metadata filtering, chunking strategies, and agent workflows) will be measured against this foundation.

---

## Future Directions

Later phases introduce:

- Agent orchestration with LangGraph  
- Observability and tracing  
- Evaluation harnesses  
- Human-in-the-loop patterns  
- Broader AI system design patterns  

---

## Who This Is For

This repository is useful for:

- Senior AI Engineers and AI Platform / RAG teams who are designing, evaluating, and shipping production-grade LLM systems  
- Technical recruiters and hiring managers looking for clear evidence of real system design depth, evaluation practices, and production thinking  
- Engineers who want to move beyond basic demos and see practical architectural decisions around retrieval, grounding, control flow, and observability

