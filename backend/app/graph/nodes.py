from typing import List, Literal
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from app.config import settings
from app.graph.state import State
from app.graph.tools import zhipu_web_search
from app.rag.vectorstore import get_vectorstore


logger = logging.getLogger("chat_flow")

llm = ChatDeepSeek(
    model=settings.deepseek_model,
    api_key=settings.deepseek_api_key,
    temperature=0.3,
)


def _ensure_messages(state: State) -> List[HumanMessage | AIMessage]:
    return list(state.get("messages", []))


class RouteDecision(BaseModel):
    """Structured output for router: only one of three values, no extra text."""

    route: Literal["DIRECT_GENERATE", "KNOWLEDGE", "WEB_SEARCH"]


def _heuristic_route(content: str) -> str:
    """Fallback keyword-based routing used when LLM routing is unavailable or uncertain."""

    route = "direct_generate"
    if any(k in content for k in ["最新", "现在", "今年", "新闻", "实时", "今天"]):
        route = "web_search"
    elif any(k in content for k in ["文档", "知识库", "资料", "说明"]):
        route = "retrieve"
    return route


def router_node(state: State) -> State:
    """Route the query to direct generation, knowledge base, or web search."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    content = (last.content if isinstance(last, HumanMessage) else "") if last else ""

    # 默认走关键字路由，防止大模型调用失败时崩溃
    route = _heuristic_route(content)
    decision_raw = "HEURISTIC"

    try:
        # 用大模型进行语义级路由判断，结构化输出避免多余解释文字
        system_msg = (
            "你是一个对话路由器，根据用户问题决定处理路径。\n"
            "只返回以下三者之一：DIRECT_GENERATE（简单闲聊/主观）、KNOWLEDGE（需查本地文档/知识库）、WEB_SEARCH（需最新/实时信息）。"
        )
        conv: list[tuple[str, str]] = [("system", system_msg)]
        for m in messages[-4:]:
            if isinstance(m, HumanMessage):
                conv.append(("human", m.content))
            elif isinstance(m, AIMessage):
                conv.append(("ai", m.content))

        structured_llm = llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke(conv)
        decision_raw = decision.route

        if decision_raw == "KNOWLEDGE":
            route = "retrieve"
        elif decision_raw == "WEB_SEARCH":
            route = "web_search"
        elif decision_raw == "DIRECT_GENERATE":
            route = "direct_generate"
        else:
            logger.warning(
                "router_node: unexpected LLM decision '%s', fallback to heuristic",
                decision_raw,
            )
            route = _heuristic_route(content)
    except Exception as exc:  # pragma: no cover - 网络/调用异常
        logger.warning("router_node: LLM routing failed: %s, fallback to heuristic", exc)
        route = _heuristic_route(content)

    if route == "retrieve":
        logger.info(
            "router_node: 去知识库中寻找（route=%s, decision=%s），query=%s",
            route,
            decision_raw,
            content[:200],
        )
    elif route == "web_search":
        logger.info(
            "router_node: 利用智谱搜索这种类似的信息（route=%s, decision=%s），query=%s",
            route,
            decision_raw,
            content[:200],
        )
    else:
        logger.info(
            "router_node: 直接生成回答（route=%s, decision=%s），query=%s",
            route,
            decision_raw,
            content[:200],
        )

    return {
        **state,
        "route": route,
        "current_node": "router",
    }


def analyze_query_node(state: State) -> State:
    """
    Before retrieve or web_search: use LLM to analyze user input and produce
    a concrete search query (what to search for).
    """
    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    user_content = (last.content if isinstance(last, HumanMessage) else "") if last else ""
    route = state.get("route") or ""

    if route == "retrieve":
        prompt = (
            "用户问题需要查本地知识库。请根据下面用户输入，提炼出 1 条最适合做向量检索的查询（关键词或短句）。"
            "不要解释，只输出这一条查询内容。\n\n用户输入：\n" + user_content
        )
        logger.info("analyze_query_node: 分析知识库检索意图，user=%s", user_content[:200])
    else:
        # web_search
        prompt = (
            "用户问题需要实时/网页搜索。请根据下面用户输入，提炼出 1 条最适合做网页搜索的查询（关键词或短句）。"
            "不要解释，只输出这一条查询内容。\n\n用户输入：\n" + user_content
        )
        logger.info("analyze_query_node: 分析网页搜索意图，user=%s", user_content[:200])

    search_query = llm.invoke([("human", prompt)]).content.strip() if prompt else user_content
    if not search_query:
        search_query = user_content

    logger.info("analyze_query_node: search_query=%s", search_query[:200])

    return {
        **state,
        "search_query": search_query,
        "current_node": "analyze_query",
    }


def retrieve_node(state: State) -> State:
    """Retrieve relevant chunks from the local vector store."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    raw = (last.content if isinstance(last, HumanMessage) else "") if last else ""
    query = (state.get("search_query") or raw).strip()

    logger.info("retrieve_node: query=%s", query[:200])

    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=5)
    retrieved = [d.page_content for d in docs]

    logger.info("retrieve_node: retrieved=%d docs", len(retrieved))

    return {
        **state,
        "retrieved_docs": retrieved,
        "current_node": "retrieve",
    }


def web_search_node(state: State) -> State:
    """Call Zhipu Web Search to get real-time information."""

    messages = _ensure_messages(state)
    last = messages[-1] if messages else None
    raw = (last.content if isinstance(last, HumanMessage) else "") if last else ""
    query = (state.get("search_query") or raw).strip()

    logger.info("web_search_node: query=%s", query[:200])

    results = zhipu_web_search(query, max_results=5)
    logger.info("web_search_node: results=%d", len(results))
    return {
        **state,
        "search_results": results,
        "current_node": "web_search",
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

    if route == "generate":
        logger.info(
            "grade_node: 找到了足够的上下文，将进入生成回答流程（知识库片段=%d，搜索结果=%d）",
            len(retrieved),
            len(search_results),
        )
    else:
        logger.info(
            "grade_node: 上下文不足，再利用智谱搜索这种类似的信息（知识库片段=%d，搜索结果=%d）",
            len(retrieved),
            len(search_results),
        )

    return {
        **state,
        "route": route,
        "current_node": "grade",
    }


def generate_node(state: State) -> State:
    """Use DeepSeek to generate an answer with available context."""

    messages = _ensure_messages(state)
    retrieved = state.get("retrieved_docs") or []
    search_results = state.get("search_results") or []

    context_parts: List[str] = []
    if retrieved:
        context_parts.append(
            "【以下来自本地知识库】\n" + "\n\n".join(retrieved[:5])
        )
    if search_results:
        context_parts.append(
            "【以下来自网页搜索，每条含标题、链接与摘要】\n" + "\n\n".join(search_results[:5])
        )

    system_prefix = (
        "你是一个中文助手，可以使用知识库和实时搜索信息。\n"
        "在回答时：\n"
        "1) 优先结合提供的上下文；\n"
        "2) 若信息不足，请直说。\n"
        "3) **必须在回答末尾单独一段标注来源**：\n"
        "   - 先写「参考来源：」然后说明本回答参考了「知识库」和/或「网页搜索」；\n"
        "   - 若使用了网页搜索结果，必须列出所引用的网站，格式为「网站标题（链接）」或「[标题](链接)」，每条一行。\n"
        "   - 仅使用了知识库时只写「参考来源：知识库」；两者都用时两者都写，并列出引用的网页链接。"
    )

    if context_parts:
        system_prefix += "\n\n" + "\n\n".join(context_parts)

    full_messages = [("system", system_prefix)]
    for m in messages:
        if isinstance(m, HumanMessage):
            full_messages.append(("human", m.content))
        elif isinstance(m, AIMessage):
            full_messages.append(("ai", m.content))

    logger.info(
        "generate_node: calling DeepSeek, context_from_kb=%d, context_from_web=%d",
        len(retrieved),
        len(search_results),
    )

    ai_msg = llm.invoke(full_messages)

    logger.info("generate_node: answer_length=%d", len(ai_msg.content or ""))

    new_messages = messages + [AIMessage(content=ai_msg.content)]

    return {
        **state,
        "messages": new_messages,
        "current_node": "generate",
    }


MAX_REFLECT_LOOPS = 2


def reflect_node(state: State) -> State:
    """
    Self-reflection with loop limit: ask the model whether more information
    is needed. If yes AND we haven't exceeded MAX_REFLECT_LOOPS, route back
    to retrieval; otherwise finish.
    """

    messages = _ensure_messages(state)
    last_ai = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
    count = state.get("reflect_count", 0)

    if not last_ai:
        logger.info("reflect_node: no last_ai, end")
        return {**state, "route": "end", "current_node": "reflect"}

    if count >= MAX_REFLECT_LOOPS:
        logger.info(
            "reflect_node: reached max loops (%d/%d), forced end",
            count, MAX_REFLECT_LOOPS,
        )
        return {
            **state,
            "route": "end",
            "current_node": "reflect",
        }

    question = (
        "请你评估上一个回答是否足够完整、可靠。\n"
        "如果你觉得答案还不确定、需要更多背景信息，请仅输出：NEED_MORE_INFO\n"
        "如果你觉得答案已经足够，请仅输出：ENOUGH\n\n"
        f"上一个回答内容如下：\n{last_ai.content}"
    )

    logger.info("reflect_node: asking judgement from DeepSeek (loop %d/%d)", count + 1, MAX_REFLECT_LOOPS)
    judgement = llm.invoke([("human", question)]).content.strip().upper()

    if "NEED_MORE_INFO" in judgement:
        route = "retrieve"
        count += 1
    else:
        route = "end"

    logger.info("reflect_node: judgement=%s, route=%s, reflect_count=%d", judgement, route, count)

    return {
        **state,
        "route": route,
        "current_node": "reflect",
        "reflect_count": count,
    }

