# AI System Patterns Roadmap

This document outlines the planned evolution of the AI System Patterns repository.  
It is a living roadmap and may evolve as the system matures.

---

## Phase 1 — RAG Foundations

### Goal
Build a RAG-based AI Engineering Assistant grounded in real technical documentation.

### Focus
– Curate a high-quality documentation corpus from LangChain and OpenAI to ground the assistant in real AI engineering knowledge  
– Implement token-aware chunking that preserves semantic coherence and supports effective retrieval  
– Build an embedding and retrieval pipeline that converts documents into vectors and retrieves relevant context for queries  
– Add confidence-based refusal logic so the system declines to answer when context is insufficient  
– Create a small evaluation set to measure retrieval quality, grounding, and refusal behavior  

### Corpus
We will curate a focused set of AI engineering documentation to serve as the knowledge base that grounds retrieval and answers in authoritative technical sources.

– LangChain docs (retrieval, vector stores, splitters, RAG)  
– OpenAI docs (embeddings, function calling)  

### Out of Scope (Phase 1)
– Agents  
– LangGraph  
– LangSmith  
– Deployment infrastructure  

---

## Later Phases

### Phase 2 — Agentic Components + Orchestration
Introduce agent control flow and decision-making.

### Phase 3 — Observability + Evaluation Systems
Add tracing, richer evaluation, and system monitoring.
