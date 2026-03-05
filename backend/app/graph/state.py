from typing import List, Optional

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Shared state that flows through the LangGraph pipeline."""

    # full conversation history
    messages: List[BaseMessage]
    # retrieved text chunks from the local knowledge base
    retrieved_docs: List[str]
    # web search snippets from Zhipu Web Search
    search_results: List[str]
    # routing decision made by the router node
    route: Optional[str]
    # name of the node that was most recently executed
    current_node: Optional[str]
    # how many times reflect has triggered a re-retrieval loop
    reflect_count: int
    # analyzed search query (filled by analyze_query node before retrieve/web_search)
    search_query: Optional[str]


State = GraphState

