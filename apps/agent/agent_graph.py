"""
Minimal LangGraph agent for orchestrating a RAG microservice.

Design goals (v1):
- Thin orchestration layer over a dedicated RAG API
- Always route domain questions to retrieval
- Preserve RAG refusal signals
- Keep logic simple and explicit

This agent is intentionally:
- Single-turn
- Stateless
- Deterministic
- Easy to inspect and evaluate

Future versions may add:
- Structured state
- Retry/reformulation on refusal
- Conditional routing


TODO (Agent v2 Roadmap)
- Add structured rag_result to state
- Add retry_count
- Add single retry edge on refusal
- Add query enhancement node

"""

from typing import TypedDict, Annotated, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .rag_query_tool import rag_query_tool


# -------------------------
# State
# -------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    rag_result: dict | None
    retry_count: int

# -------------------------
# Model + Tool Binding
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

llm_with_tools = llm.bind_tools([rag_query_tool])


# -------------------------
# Nodes
# -------------------------

def agent_node(state: AgentState):
    """
    Force tool use for domain questions.
    """

    system_prompt = SystemMessage(
        content=(
            "You are a precise AI Engineering Assistant focused on "
            "LangChain and RAG systems.\n"
            "For any question in this domain, you MUST call the "
            "rag_query_tool.\n"
            "Never answer from general knowledge.\n"
            "Respect the tool's refusal logic exactly."
        )
    )

    response = llm_with_tools.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response],
        "rag_result": state.get("rag_result"),   # NEW: preserve structured result
        "retry_count": state.get("retry_count", 0),  # NEW: preserve retry count
    }


def tool_node(state: AgentState):
    """
    Execute rag_query_tool and return clean output.
    """

    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls or []

    if not tool_calls:
        return {
            "messages": [
                AIMessage(
                    content="I cannot answer this query."
                )
            ],
            "rag_result": None,  # NEW: explicitly return structured state
            "retry_count": state.get("retry_count", 0),  # NEW
        }

    results = []
    final_result: dict[str, Any] | None = None  # FIX: initialize before loop

    for call in tool_calls:
        if call["name"] == "rag_query_tool":

            result: dict[str, Any] = rag_query_tool.invoke(
                call["args"]
            )
            
            final_result = result  # NEW: capture structured result

            if result.get("refused", False):
                content = (
                    f"{result.get('answer')}\n\n"
                    f"(Refusal reason: "
                    f"{result.get('refusal_reason', 'unknown')})"
                )
            else:
                content = result.get(
                    "answer",
                    "No answer returned."
                )

            results.append(
                AIMessage(content=content)
            )

    return {"messages": results,
            "rag_result": final_result,  # NEW: store structured RAG output
            "retry_count": state.get("retry_count", 0),  # NEW
            }


# -------------------------
# Graph
# -------------------------

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

# Always call tool in v0
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

agent = graph.compile()
