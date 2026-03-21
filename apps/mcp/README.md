# MCP Integration

This module implements a Model Context Protocol (MCP)-style interface for exposing system capabilities as callable tools.

The goal is to provide a clean abstraction layer where components like retrieval and agent execution can be accessed programmatically by external systems.

This mirrors modern AI system design, where capabilities are exposed as tools rather than hardcoded workflows.

---

## Architecture

The MCP layer is structured with clear separation of concerns:

- `mcp_tool_specs.py` → tool definitions (names, schemas)
- `mcp_tool_handlers.py` → tool implementations
- `mcp_server.py` → tool registry and dispatch (`call_tool`)
- `mcp_api.py` → HTTP interface for external access

Flow:

external request → MCP API → call_tool → handler → system logic (RAG / agent)

---

## MCP API

The system exposes tools via a simple HTTP interface.

### Endpoint

POST `/mcp/call_tool`

### Request

```json
{
  "name": "rag_query",
  "args": {
    "query": "What is LangChain?"
  }
}
```

### Example
```bash
curl -X POST http://localhost:8001/mcp/call_tool \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag_query",
    "args": {
      "query": "What is LangChain?"
    }
  }'
```

### Response

Returns structured output from the tool, including answer, refusal status, and retrieval diagnostics.

---

## Available Tools
`rag_query` — query the RAG pipeline via the MCP interface

Additional tools will be added to expose more system capabilities.

---

## Design Goals
- Expose system capabilities as reusable tools
- Provide a unified interface layer for external clients
- Maintain clean separation between interface, orchestration, and core logic
- Enable extensibility (new tools can be added without changing the interface)

---

## Next Steps
- Route agent execution through MCP tools
- Add agent_run tool
- Expand tool set (evaluation, diagnostics)
- Integrate with external MCP-compatible clients