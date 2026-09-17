from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents.research_agent import research_agent
from app.agents.summary_agent import summary_agent
from app.agents.comparison_agent import comparison_agent


# ============================================================
# 1. GRAPH STATE
# ============================================================

class AgentState(TypedDict):
    query: str
    answer: str
    sources: list
    agent: str


# ============================================================
# 2. HYBRID SUPERVISOR
# ============================================================

def supervisor_node(state: AgentState):

    query = state["query"].lower().strip()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_keywords = [
        "summarize",
        "summary",
        "overview",
        "give me an overview",
        "summarise",
        "brief summary",
        "short summary",
    ]

    if any(keyword in query for keyword in summary_keywords):

        decision = "SUMMARY"

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    elif (
        "compare" in query
        or "comparison" in query
        or "differences between" in query
        or "difference between" in query
        or "similarities between" in query
    ):

        decision = "COMPARISON"

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    else:

        decision = "RESEARCH"

    print(f"\nSupervisor selected: {decision}")

    return {
        "agent": decision
    }


# ============================================================
# 3. RESEARCH NODE
# ============================================================

def research_node(state: AgentState):

    query = state["query"]

    result = research_agent(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "agent": "RESEARCH"
    }


# ============================================================
# 4. SUMMARY NODE
# ============================================================

def summary_node(state: AgentState):

    query = state["query"]

    result = summary_agent(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "agent": "SUMMARY"
    }

# ============================================================
# 5. COMPARISON NODE
# ============================================================

def comparison_node(state: AgentState):

    query = state["query"]

    result = comparison_agent(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "agent": "COMPARISON"
    }

# ============================================================
# 6. ROUTING FUNCTION
# ============================================================

def route_agent(state: AgentState):

    agent = state["agent"]

    if agent == "SUMMARY":
        return "summary"

    if agent == "COMPARISON":
        return "comparison"

    return "research"


# ============================================================
# 7. BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("summary", summary_node)
    graph.add_node("comparison", comparison_node)
    # START → SUPERVISOR
    graph.add_edge(START, "supervisor")

    # SUPERVISOR → AGENT
    graph.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "research": "research",
            "summary": "summary",
            "comparison": "comparison",
        }
    )

    # AGENT → END
    graph.add_edge("research", END)
    graph.add_edge("summary", END)
    graph.add_edge("comparison", END)
    return graph.compile()


# ============================================================
# 8. COMPILED GRAPH
# ============================================================

research_graph = build_graph()