# MCP Integration

This directory will contain the Model Context Protocol (MCP) integration for the AI Engineering Patterns project.

The goal of this module is to expose parts of the system, such as the RAG pipeline, as MCP tools that can be used by external LLM clients and agent frameworks.

## Planned Components

- MCP server exposing system capabilities
- Tool definitions for agent interaction
- Example MCP client usage
- Documentation for running the MCP server locally

This will demonstrate how an AI application can expose structured capabilities through MCP.

## Initial Tools

The MCP server will expose internal capabilities of the system as tools.

Planned tools:

- `rag_query` — query the RAG pipeline
- `vector_search` — direct retrieval queries
- `eval_status` — inspect recent evaluation runs