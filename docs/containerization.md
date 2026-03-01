# Containerization Strategy

## Overview

This project uses a multi-container architecture:

- `rag-api` → FastAPI service exposing `/query`
- `agent` → CLI agent that calls the RAG API
- Docker Compose manages networking between services

## Design Principles

- Services are isolated by runtime concern
- Internal Docker DNS is used for service-to-service communication
- `.dockerignore` minimizes build context size
- Environment variables configure service endpoints
- The agent defaults to CLI mode but can be run in eval mode

## Networking

Inside Docker:

```
agent → http://rag-api:8000/query
```

No host bridge required.

## Build Hygiene

`.dockerignore` excludes:

- `.venv`
- `data/vectorstore`
- `logs`
- `.git`
- etc.

This keeps images small and builds fast.

## Running Locally

Start RAG only:

```bash
docker compose up -d rag-api
```

Run agent interactively (after starting `rag-api` via compose):

```bash
docker run --rm -it \
  --network ai-engineering-patterns_default \
  --env-file .env \
  -e RAG_API_URL=http://rag-api:8000/query \
  agent:dev
```

Stop everything:
```bash
docker compose down
```