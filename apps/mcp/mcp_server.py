"""
MCP server for the AI Engineering Patterns project.

This server exposes system capabilities as tools that can be
discovered and invoked dynamically.

Tool specifications are defined in `mcp_tool_specs.py`.
Tool implementations are defined in `mcp_tool_handlers.py`.
"""

from apps.mcp.mcp_tool_specs import MCP_TOOL_SPECS
from apps.mcp import mcp_tool_handlers as handlers


# -------------------------
# Tool registry
# -------------------------
TOOL_REGISTRY = {
    "rag_query": handlers.rag_query,
    "vector_search": handlers.vector_search,
    "eval_status": handlers.eval_status,
}


def get_tool_specs():
    """
    Return the MCP tool specifications exposed by this server.
    """
    return MCP_TOOL_SPECS


def call_tool(name: str, args: dict):
    """
    Invoke a tool by name with arguments.
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    handler = TOOL_REGISTRY[name]
    return handler(**args)


# -------------------------
# Local test harness
# -------------------------
def main():
    tools = get_tool_specs()

    print("Available MCP tools:")
    for tool in tools:
        print(f"- {tool['name']}")

    print("\nTesting rag_query...\n")

    result = call_tool("rag_query", {"query": "What is LangChain?"})
    print(result["answer"])


if __name__ == "__main__":
    main()