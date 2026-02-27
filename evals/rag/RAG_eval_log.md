# RAG Evaluation Log

## Eval Run v1 — Baseline

Date: 2026-02-07  
Dataset: rag_eval_queries_v1.json  
k: 5  
MAX_DISTANCE: 1.2  

Results:
- Retrieval hit rate: 10/16 (62.5%)
- Correct refusals: 0/3
- Overall pass rate: 62.5%

Observations:
- Definition queries perform well
- Conceptual linkage queries weaker
- Refusal detection too permissive
- Common distractor: philosophy.mdx

Next step:
- Lower MAX_DISTANCE to improve refusal behavior


## Eval Run v2

Date: 2026-02-07  
Dataset: rag_eval_queries_v1.json  
k: 5  
MAX_DISTANCE: 0.95

Results:
- Retrieval hit rate: 7/16 (43.8%)
- Correct refusals: 3/3
- Overall pass rate: 62.5%

Observations:
- Recall much worse than for MAX_DISTANCE: 1.2  
- Conceptual linkage queries weaker
- Much better at refusals than for MAX_DISTANCE: 1.2  

Next step:
- Increase MAX_DISTANCE slightly to maintain refusal behavior but increase hit rate


## Eval Run v3

Date: 2026-02-07  
Dataset: rag_eval_queries_v1.json  
k: 5  
MAX_DISTANCE: 1.05  

Results:
- Retrieval hit rate: 10/16 (62.5%)
- Correct refusals: 1/3 (33.3%)
- Overall pass rate: 11/16 (68.8%)

Observations:
- MAX_DISTANCE = 1.05 represents a middle ground between recall and precision compared to earlier runs.
- Retrieval quality is solid for core RAG concepts (retrievers, RAG pipeline, embeddings, memory, MCP).
- Failures cluster around:
  - Knowledge base queries
  - Observability queries
  - Opinion-based or out-of-scope questions
- Some near-miss failures retrieve semantically related but non-authoritative docs (e.g., philosophy.mdx, overview.mdx), suggesting embedding similarity without topical specificity.
- Refusal behavior improved compared to the most permissive threshold, but the system still retrieves marginally relevant docs for opinion-style queries (“best vector store,” pricing questions).

Takeaway:
- The system is directionally correct but still biased slightly toward recall over strict precision.
- This is acceptable for a v1 demo system, but not ideal for high-stakes or compliance-heavy use cases.

Next step:
- Keep MAX_DISTANCE = 1.05 as the v1 baseline.
- Improve precision via:
  - Metadata filtering (e.g., prioritize conceptual docs over install/help docs)
  - Better chunking strategy
  - Optional reranking stage
- In v2, add a reranker or hybrid retrieval (BM25 + vector) to reduce topical drift.