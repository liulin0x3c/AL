from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from app.config import settings
from app.graph.state import State
from app.graph.tools import zhipu_web_search
from app.rag.vectorstore import get_vectorstore


llm = ChatDeepSeek(
    model=settings.deepseek_model,
    api_key=settings.deepseek_api_key,
    temperature=0.3,
)


def _ensure_messages(state: State) -> List[HumanMessage | AIMessage]:
    return list(state.get("messages", []))


def router_node(state: State) -> State:
    """Route the query to direct generation, knowledge base, or web search."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    content = (last.content if isinstance(last, HumanMessage) else "") if last else ""

    route = "direct_generate"
    if any(k in content for k in ["最新", "现在", "今年", "新闻", "实时", "今天"]):
        route = "web_search"
    elif any(k in content for k in ["文档", "知识库", "资料", "说明"]):
        route = "retrieve"

    return {
        **state,
        "route": route,
    }


def retrieve_node(state: State) -> State:
    """Retrieve relevant chunks from the local vector store."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    query = (last.content if isinstance(last, HumanMessage) else "") if last else ""

    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=5)
    retrieved = [d.page_content for d in docs]

    return {
        **state,
        "retrieved_docs": retrieved,
    }


def web_search_node(state: State) -> State:
    """Call Zhipu Web Search to get real-time information."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    query = (last.content if isinstance(last, HumanMessage) else "") if last else ""

    results = zhipu_web_search(query, max_results=5)
    return {
        **state,
        "search_results": results,
    }


def grade_node(state: State) -> State:
    """
    Simple grading: if we have any retrieved docs or search results, we proceed
    to generation; otherwise, we try web search as a fallback.
    """

    retrieved = state.get("retrieved_docs") or []
    search_results = state.get("search_results") or []

    if retrieved or search_results:
        route = "generate"
    else:
        route = "web_search"

    return {
        **state,
        "route": route,
    }


def generate_node(state: State) -> State:
    """Use DeepSeek to generate an answer with available context."""

    messages = _ensure_messages(state)
    retrieved = state.get("retrieved_docs") or []
    search_results = state.get("search_results") or []

    context_parts: List[str] = []
    if retrieved:
        context_parts.append(
            "以下是来自本地知识库的相关内容片段：\n" + "\n\n".join(retrieved[:5])
        )
    if search_results:
        context_parts.append(
            "以下是来自实时网页搜索的结果片段：\n" + "\n\n".join(search_results[:5])
        )

    system_prefix = (
        "你是一个中文助手，可以使用知识库和实时搜索信息。\n"
        "在回答时：\n"
        "1) 优先结合提供的上下文；\n"
        "2) 明确区分本地知识库与实时搜索的来源；\n"
        "3) 如果信息不足，请直说。"
    )

    if context_parts:
        system_prefix += "\n\n" + "\n\n".join(context_parts)

    full_messages = [("system", system_prefix)]
    for m in messages:
        if isinstance(m, HumanMessage):
            full_messages.append(("human", m.content))
        elif isinstance(m, AIMessage):
            full_messages.append(("ai", m.content))

    ai_msg = llm.invoke(full_messages)

    new_messages = messages + [AIMessage(content=ai_msg.content)]

    return {
        **state,
        "messages": new_messages,
    }


def reflect_node(state: State) -> State:
    """
    Simple self-reflection: ask the model whether more information is needed.
    If the answer clearly states it's uncertain / needs more info, we will
    route back to retrieval; otherwise we finish.
    """

    messages = _ensure_messages(state)
    last_ai = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    if not last_ai:
        return {**state, "route": "end"}

    question = (
        "请你评估上一个回答是否足够完整、可靠。\n"
        "如果你觉得答案还不确定、需要更多背景信息，请仅输出：NEED_MORE_INFO\n"
        "如果你觉得答案已经足够，请仅输出：ENOUGH\n\n"
        f"上一个回答内容如下：\n{last_ai.content}"
    )

    judgement = llm.invoke([("human", question)]).content.strip().upper()

    if "NEED_MORE_INFO" in judgement:
        route = "retrieve"
    else:
        route = "end"

    return {
        **state,
        "route": route,
    }

