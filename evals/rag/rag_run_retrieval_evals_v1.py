"""
Offline retrieval evaluation for the RAG vector store.

This script tests the retrieval layer in isolation by:
- Running evaluation queries against the persisted Chroma vector DB
- Filtering results by similarity distance threshold
- Comparing retrieved document sources to expected sources

Purpose:
- Diagnose retrieval failures independently of generation logic
- Tune distance threshold (MAX_DISTANCE)
- Measure retrieval hit rate for in-scope queries
- Identify cases that should produce refusal due to missing context

This evaluation is run before:
- API-level RAG evals
- Agent-level reasoning evals

... to ensure retrieval quality is not the primary source of downstream errors.
"""

import json
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

EVAL_FILE = BASE_DIR / "evals/rag/rag_eval_queries_v1.json"
PERSIST_DIR = BASE_DIR / "data/vectorstore/langchain_db"


# -------------------------
# Config
# -------------------------
K = 5
MAX_DISTANCE = 1.05  # retrieval relevance threshold


# -------------------------
# Retrieval helper
# -------------------------
def retrieve(
    vectordb: Chroma,
    query: str,
    k: int = K,
) -> List[Tuple[Document, float]]:
    results = vectordb.similarity_search_with_score(query, k=k)
    filtered = [
        (document, distance)
        for document, distance in results
        if distance <= MAX_DISTANCE
    ]
    return filtered


# -------------------------
# Eval logic
# -------------------------
def main():

    # Load eval dataset
    with open(EVAL_FILE, "r") as eval_file:
        eval_data = json.load(eval_file)

    # Load vector DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    print(f"\nCollection count: {vectordb._collection.count()}\n")

    total = len(eval_data)
    in_scope_total = 0
    retrieval_hits = 0
    correct_refusals = 0
    passes = 0

    print("==== Running RAG Eval v1 ====\n")

    # -------------------------
    # Main evaluation loop
    # -------------------------
    for eval_case in eval_data:

        eval_id = eval_case["id"]
        eval_query = eval_case["query"]
        expected_sources = set(eval_case["expected_sources"])
        must_refuse = eval_case["must_refuse"]

        if not must_refuse:
            in_scope_total += 1

        print(f"--- {eval_id} ---")
        print(f"Q: {eval_query}")

        # Run similarity search + distance filtering
        results = retrieve(vectordb, eval_query)

        # -------------------------
        # Case 1: No relevant retrieval
        # -------------------------
        if not results:
            print("No relevant retrieval.\n")

            if must_refuse:
                correct_refusals += 1
                passes += 1
                print("✓ Correct refusal expected\n")
            else:
                print("✗ Retrieval miss\n")

            continue

        # -------------------------
        # Case 2: Relevant docs retrieved
        # -------------------------
        retrieved_sources = {
            Path(doc.metadata.get("source", "")).name
            for doc, _ in results
        }

        print("Retrieved sources:", retrieved_sources)

        # Retrieval "hit" occurs if any expected source is found
        hit = bool(expected_sources & retrieved_sources)

        # -------------------------
        # Pass / fail logic
        # -------------------------
        if must_refuse:
            print("✗ Should have refused\n")
        else:
            if hit:
                retrieval_hits += 1
                passes += 1
                print("✓ Retrieval hit\n")
            else:
                print("✗ Retrieved but wrong docs\n")

    # -------------------------
    # Summary
    # -------------------------
    hit_rate = retrieval_hits / in_scope_total if in_scope_total > 0 else 0

    print("\n==== Summary ====")
    print(f"Total queries: {total}")
    print(f"In-scope queries: {in_scope_total}")
    print(f"Retrieval hit rate: {retrieval_hits}/{in_scope_total} ({hit_rate:.2%})")
    print(f"Correct refusals: {correct_refusals}")
    print(f"Overall passes: {passes}/{total}")
    print("==================\n")


if __name__ == "__main__":
    main()
