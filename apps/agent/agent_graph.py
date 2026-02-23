"""
LangGraph agent for orchestrating a RAG microservice with controlled escalation.

Design goals (v2):
- Thin orchestration layer over a dedicated RAG API
- Preserve structured RAG refusal signals
- Deterministic, bounded retry logic
- Explicit state transitions for inspection and evaluation
- Controlled query enhancement on failure

Architecture:
- Single-turn
- Deterministic escalation ladder:
    0 → Raw query
    1 → Single rewrite (LLM reformulation)
    2 → Multi-query expansion
- Maximum three retrieval attempts
- No infinite loops
- Refusal-safe (no hallucination fallback)

State includes:
- original_query
- current_query
- rag_result (structured tool output)
- escalation_level (0,1,2)
- retry_count

This design prioritizes:
- Grounded answers
- Transparent control flow
- Evaluability
- Production safety

Further enhancements (future versions):
- Observability hooks (LangSmith)
- Metrics logging
- Dynamic escalation policies
- Tool-aware agent routing
"""

from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import  HumanMessage, AIMessage

from .rag_query_tool import rag_query_tool


# -------------------------
# State
# -------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    current_query: str
    rag_result: dict | None
    escalation_level: int  # 0=raw,1=rewrite,2=multi
    retry_count: int


# ------
# Model 
# ------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)



# -------------------------
# Nodes
# -------------------------

def agent_node(state: AgentState):
    """
    Initialize structured state.
    """
    user_message = state["messages"][-1]

    return {
        "messages": [],
        "original_query": user_message.content,
        "current_query": user_message.content,
        "rag_result": None,
        "escalation_level": 0,
        "retry_count": 0,
    }

def run_rag_node(state: AgentState):
    """
    Call rag_query_tool using current_query.
    """

    result = rag_query_tool.invoke({
        "query": state["current_query"]
    })

    if result.get("refused", False):
        content = (
            f"{result.get('answer')}\n\n"
            f"(Refusal reason: "
            f"{result.get('refusal_reason', 'unknown')})"
        )
    else:
        content = result.get("answer", "No answer returned.")

    return {
        "messages": [AIMessage(content=content)],
        "rag_result": result,
        "original_query": state["original_query"],
        "current_query": state["current_query"],
        "escalation_level": state["escalation_level"],
        "retry_count": state["retry_count"],
    }


def rewrite_node(state: AgentState):
    """
    Rewrite original query once.
    """

    prompt = (
        "Rewrite the following query to improve retrieval quality. "
        "Preserve meaning. Do not add new facts.\n\n"
        f"Query:\n{state['original_query']}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [],
        "original_query": state["original_query"],
        "current_query": response.content.strip(),
        "rag_result": state["rag_result"],
        "escalation_level": 1,
        "retry_count": state["retry_count"] + 1,
    }


def multi_query_node(state: AgentState):
    """
    Generate alternate phrasings and concatenate.
    """

    prompt = (
        "Generate 2 alternative phrasings of the following query "
        "to improve semantic retrieval. Separate each on a newline.\n\n"
        f"Query:\n{state['original_query']}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    alternates = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    combined_query = "\n".join(
        [state["original_query"]] + alternates
    )

    return {
        "messages": [],
        "original_query": state["original_query"],
        "current_query": combined_query,
        "rag_result": state["rag_result"],
        "escalation_level": 2,
        "retry_count": state["retry_count"] + 1,
    }

def decision_node(state: AgentState):
    """
    Decide next step based on refusal + escalation level.
    """

    result = state.get("rag_result")

    if result and not result.get("refused", False):
        return "end"

    if state["escalation_level"] == 0:
        return "rewrite"

    if state["escalation_level"] == 1:
        return "multi"

    return "end"



# -------------------------
# Graph
# -------------------------

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("run_rag", run_rag_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("multi", multi_query_node)


graph.set_entry_point("agent")

graph.add_edge("agent", "run_rag")
graph.add_conditional_edges(
    "run_rag",
    decision_node,
    {
        "rewrite": "rewrite",
        "multi": "multi",
        "end": END,
    },
)


graph.add_edge("rewrite", "run_rag")
graph.add_edge("multi", "run_rag")

agent = graph.compile()
