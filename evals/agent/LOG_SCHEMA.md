# Agent Evaluation Log Schema
This schema defines the structure of JSONL logs produced by `agent_run_evals_v1.py`.

## Current Version: 2

Each JSONL row contains:

- eval_schema_version (int): schema version of the log format
- run_id (string): unique identifier for a single evaluation run
- agent_version (string): version label of the agent system (e.g., graph / orchestration logic)
- rag_version (string): version label of the retrieval system used by the agent
- model (string): LLM model used during the run
- git_commit (string): git commit hash at time of execution
- ts (float): UNIX timestamp when the eval case was executed
- eval_id (string): identifier of the eval case
- query (string): input query for the eval case
- category (string): eval classification (e.g., in_scope, borderline, out_of_scope)
- answer (string): final agent response
- must_refuse (bool): whether the eval expects a refusal
- refused (bool): whether the agent produced a refusal
- passed (bool): whether the eval case passed
- latency_sec (float): end-to-end latency in seconds

## Version History

### v1 (deprecated)
- No run-level metadata
- No schema version field
- Not reproducible

### v2
- Added run-level metadata fields
- Added eval_schema_version
- Enables reproducibility and historical comparison

## Migration Policy

- Historical logs may be migrated forward
- Original files are preserved or archived
- Migration scripts must be non-destructive
- Migration utilities are located in `scripts/`