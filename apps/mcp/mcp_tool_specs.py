"""
MCP tool specifications for the AI Engineering Patterns project.

This module defines the tools that the MCP server will expose to external
LLM clients and agent frameworks.

Each tool specification includes:

- name
- description
- input schema (JSON Schema style)

These specifications describe the interface contract for MCP tools.
Actual tool execution logic will be implemented separately.
"""

MCP_TOOL_SPECS = [
    {
        "name": "rag_query",
        "description": "Query the RAG pipeline and return a grounded answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query to send to the RAG pipeline."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "vector_search",
        "description": "Run a direct similarity search against the vector store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for vector similarity."
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "eval_status",
        "description": "Return information about the most recent evaluation runs.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]
