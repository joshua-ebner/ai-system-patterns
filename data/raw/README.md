# Raw Data

This folder contains **unmodified documentation** pulled from upstream sources.

Examples:
- LangChain docs (MD/MDX from GitHub)
- OpenAI docs

No cleaning, parsing, or chunking is performed here.

---

## Rebuilding This Folder

Raw docs are **not committed to the repo** to avoid bloat and licensing issues.

To pull the data:

```bash
bash scripts/pull_langchain_docs.sh
