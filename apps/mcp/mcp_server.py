"""
MCP server for the AI Engineering Patterns project.

This server will expose selected system capabilities through the
Model Context Protocol (MCP), allowing external LLM clients and
agent frameworks to discover and invoke tools.

Tool specifications are defined in `mcp_tool_specs.py`.
"""

from apps.mcp.mcp_tool_specs import MCP_TOOL_SPECS


def get_tool_specs():
    """
    Return the MCP tool specifications exposed by this server.
    """
    return MCP_TOOL_SPECS


def main():
    """
    Temporary entrypoint for verifying tool registration.
    Later this will start the MCP server.
    """
    tools = get_tool_specs()

    print("Available MCP tools:")
    for tool in tools:
        print(f"- {tool['name']}")


if __name__ == "__main__":
    main()