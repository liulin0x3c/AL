from typing import List

from zhipuai import ZhipuAI

from app.config import settings


def zhipu_web_search(query: str, max_results: int = 5) -> List[str]:
    """
    Call Zhipu Web Search via the `web-search-pro` model.

    Note: The exact response schema may evolve; this function is written
    defensively and will fall back to a simple error message if parsing fails.
    """

    client = ZhipuAI(
        api_key=settings.zhipuai_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    try:
        resp = client.chat.completions.create(
            model="web-search-pro",
            messages=[{"role": "user", "content": query}],
            stream=False,
        )
    except Exception as exc:  # pragma: no cover - network error path
        return [f"Web search failed: {exc}"]

    results: List[str] = []

    try:
        choice = resp.choices[0]
        # In many examples, web search data is exposed via tool_calls/search_result.
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        for tool_call in tool_calls:
            search_result = getattr(tool_call, "search_result", None)
            if not search_result:
                continue
            for item in search_result[:max_results]:
                title = item.get("title") or ""
                url = item.get("url") or item.get("link") or ""
                content = item.get("content") or item.get("summary") or ""
                snippet = f"标题: {title}\n链接: {url}\n摘要: {content}"
                results.append(snippet)
        if not results and getattr(choice.message, "content", None):
            # Fallback: some deployments may directly answer in content
            results.append(str(choice.message.content))
    except Exception as exc:  # pragma: no cover - schema drift path
        results = [f"Web search parsing failed: {exc}"]

    return results

