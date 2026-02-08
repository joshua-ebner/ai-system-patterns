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


