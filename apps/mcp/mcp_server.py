"""
MCP server for the AI Engineering Patterns project.

This module will expose selected system capabilities through the
Model Context Protocol (MCP), allowing external LLM clients and
agent frameworks to interact with the application through
structured tool calls.

Initial capabilities planned for exposure:

- rag_query
    Query the RAG pipeline and return grounded answers.

- vector_search
    Perform direct retrieval queries against the vector database.

- eval_status
    Inspect recent evaluation runs and system diagnostics.

The MCP server will act as a thin interface layer over existing
application components, enabling agent frameworks to discover
and invoke tools in a standardized way.

Implementation to follow.
"""