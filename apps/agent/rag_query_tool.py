"""
LangChain tool wrapper for the RAG FastAPI service (`apps/rag/rag_api`).

This module exposes the retrieval system as a LangChain tool so it can be
used by a LangGraph agent as an external capability rather than embedding
retrieval logic directly in the agent.

Design goals:
- Keep retrieval as a separate microservice boundary
- Provide a thin, reliable HTTP wrapper over the RAG API
- Preserve structured fields like `refused`, `sources`, and
  `refusal_reason` for downstream logic and evaluation
- Enable tracing of RAG calls via LangSmith

This module intentionally contains minimal logic. All retrieval,
grounding, and refusal decisions are delegated to the RAG service.
"""


from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict
import os
import httpx
from langchain_core.tools import tool
from langsmith import traceable



# Configurable endpoint
RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/query"
)


@traceable(name="rag_api_http_call")
def _call_rag_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Low-level HTTP call to RAG API (traced)."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(RAG_API_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


@tool
def rag_query_tool(query: str) -> Dict[str, Any]:
    """
    Query the RAG API for grounded answers from AI engineering documentation.
    """

    payload = {"query": query.strip()}

    try:
        return _call_rag_api(payload)

    except Exception as e:
        return {
            "answer": "RAG service unavailable.",
            "refused": True,
            "sources": [],
            "refusal_reason": "tool_error",
            "error": str(e),
        }
