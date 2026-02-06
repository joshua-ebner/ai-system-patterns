# Processed Data

This folder contains chunked and normalized documentation
used as the knowledge base for the RAG system.

Data here is derived from `data/raw/` and prepared for embedding and retrieval.

---

## What Lives Here

Typical contents:

- Chunked text
- Source metadata
- JSONL caches of chunks

Example:
- `langchain/chunks.jsonl` — chunked LangChain docs

---

## Data ETL Phases

Phase 1 focuses on selected LangChain and OpenAI documentation.

Later phases will likely expand this corpus with:
- LangGraph
- LangSmith
- Additional domain-specific sources

---

## How This Data Is Created

Processed data is generated from raw docs using chunking scripts.

Example:

```bash
python scripts/chunk_langchain_docs.py

