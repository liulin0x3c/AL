from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    analyze_query_node,
    generate_node,
    grade_node,
    reflect_node,
    retrieve_node,
    router_node,
    web_search_node,
)
from app.graph.state import State


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""

    workflow = StateGraph(State)

    # Core nodes
    workflow.add_node("router", router_node)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)

    # Entry: route the request
    workflow.add_edge(START, "router")

    # Router: direct -> generate; retrieve/web_search -> analyze_query first
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("route"),
        {
            "direct_generate": "generate",
            "retrieve": "analyze_query",
            "web_search": "analyze_query",
        },
    )

    # After analyze_query, go to retrieve or web_search by route
    workflow.add_conditional_edges(
        "analyze_query",
        lambda state: state.get("route"),
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        },
    )

    # After retrieve or web_search, go to grade
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("web_search", "grade")

    # Grade branching: either generate or try web_search as fallback (via analyze_query)
    workflow.add_conditional_edges(
        "grade",
        lambda state: state.get("route"),
        {
            "generate": "generate",
            "web_search": "analyze_query",
        },
    )

    # After generation, go to reflection
    workflow.add_edge("generate", "reflect")

    # Reflection: end or re-retrieve (via analyze_query to re-analyze search intent)
    workflow.add_conditional_edges(
        "reflect",
        lambda state: state.get("route"),
        {
            "end": END,
            "retrieve": "analyze_query",
        },
    )

    return workflow


graph = build_graph().compile()


def run_graph(message: str, history: list[dict] | None = None) -> list[dict]:
    """
    Helper for FastAPI: run the LangGraph flow given a user message and
    optional history in plain dict format.
    """

    messages = []
    history = history or []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=message))

    final_state = graph.invoke(
        {
            "messages": messages,
            "retrieved_docs": [],
            "search_results": [],
            "route": None,
            "current_node": None,
            "reflect_count": 0,
            "search_query": None,
        }
    )

    out_messages = final_state["messages"]
    result: list[dict] = []
    for m in out_messages:
        role = "assistant"
        if isinstance(m, HumanMessage):
            role = "user"
        result.append({"role": role, "content": m.content})
    return result

