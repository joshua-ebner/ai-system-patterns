"""
RAG FastAPI Service (v1)

This module implements a Retrieval-Augmented Generation (RAG) microservice
exposed via FastAPI.

Responsibilities:
- Accept natural-language queries via HTTP
- Retrieve relevant document chunks from a Chroma vector store
- Generate grounded answers using an LLM
- Enforce conservative refusal behavior when context is insufficient
- Log structured query/response data for evaluation and diagnostics

Design notes:
- Retrieval and generation are tightly scoped and deterministic
  (temperature=0, distance thresholding).
- The service is intentionally stateless and single-turn.
- This API is treated as a dedicated retrieval service by the
  agent layer (apps/agent), which calls it as an external tool.

This separation mirrors production architectures where retrieval
pipelines and agent orchestration are deployed as independent services.
"""


import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
PERSIST_DIR = BASE_DIR / "data/vectorstore/langchain_db"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "rag_queries_v1.jsonl"



# -------------------------
# Config
# -------------------------
K = 5
MAX_DISTANCE = 1.05

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
REFUSAL_TEXT = "I don't have enough relevant context to answer confidently."

# -------------------------
# App + globals
# -------------------------
app = FastAPI(title="RAG API v1", version="0.1.0")

_vectordb: Chroma | None = None
_llm: ChatOpenAI | None = None


# -------------------------
# Request / Response models
# -------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SourceHit(BaseModel):
    source: str
    distance: float
    snippet: str


class QueryResponse(BaseModel):
    query: str
    refused: bool
    answer: str
    sources: List[SourceHit]
    refusal_reason: str | None = None


# -------------------------
# Helpers
# -------------------------
def retrieve(
    vectordb: Chroma,
    query: str,
    k: int = K,
) -> List[Tuple[Document, float]]:
    results = vectordb.similarity_search_with_score(query, k=k)
    filtered = [(doc, dist) for doc, dist in results if dist <= MAX_DISTANCE]
    return filtered


def format_context(docs_and_scores: List[Tuple[Document, float]]) -> str:
    parts: List[str] = []

    for idx, (doc, dist) in enumerate(docs_and_scores, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(
            f"[Chunk {idx} | Source: {source} | distance={dist:.3f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


def build_sources(docs_and_scores: List[Tuple[Document, float]]) -> List[SourceHit]:
    sources: List[SourceHit] = []

    for doc, dist in docs_and_scores:
        source = Path(doc.metadata.get("source", "unknown")).name
        snippet = doc.page_content[:300].replace("\n", " ")

        sources.append(
            SourceHit(
                source=source,
                distance=float(dist),
                snippet=snippet,
            )
        )

    return sources


def log_query(payload: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup() -> None:
    global _vectordb, _llm

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    _vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    _llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
    )


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    ok = _vectordb is not None and _llm is not None
    count = _vectordb._collection.count() if _vectordb else 0

    return {
        "ok": ok,
        "collection_count": count,
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:

    if _vectordb is None or _llm is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    q = req.query.strip()
    retrieved = retrieve(_vectordb, q, k=K)

    # -------------------------
    # Refusal: no retrieval
    # -------------------------
    if not retrieved:

        latency = time.time() - start_time

        refusal_reason = "no_relevant_chunks"

        response = QueryResponse(
            query=q,
            refused=True,
            answer=REFUSAL_TEXT,
            sources=[],
            refusal_reason=refusal_reason,
        )

        log_query({
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "query": q,
            "answer": response.answer,
            "refused": True,
            "refusal_reason": refusal_reason,
            "sources": [],
            "num_chunks": 0,
            "latency_sec": latency,
            "embed_model": EMBED_MODEL,
            "llm_model": LLM_MODEL,
            "k": K,
            "max_distance": MAX_DISTANCE,
        })

        return response

    # -------------------------
    # Generation
    # -------------------------
    context = format_context(retrieved)

    prompt = f"""
You are a careful RAG assistant.

Answer the user's question using ONLY the provided context.

If the context is insufficient, respond with exactly:
"I don't have enough relevant context to answer confidently."

Question:
{q}

Context:
{context}

Answer:
""".strip()

    answer = _llm.invoke(prompt).content.strip()

    sources = build_sources(retrieved)

    refused = answer.strip() == REFUSAL_TEXT
    refusal_reason = "llm_self_refusal" if refused else None

    latency = time.time() - start_time

    response = QueryResponse(
        query=q,
        refused=refused,
        answer=answer,
        sources=sources,
        refusal_reason=refusal_reason,
    )

    log_query({
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "query": q,
        "answer": answer,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "sources": [
            {"source": s.source, "distance": s.distance}
            for s in sources
        ],
        "num_chunks": len(sources),
        "latency_sec": latency,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "k": K,
        "max_distance": MAX_DISTANCE,
    })

    return response