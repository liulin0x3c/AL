from typing import Iterable, List, Optional, Tuple, Any
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.graph.graph import graph, run_graph


router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger("chat_api")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    messages: List[Message]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Single-turn chat endpoint that runs the LangGraph flow."""

    logger.info("chat_request: message=%s, history_len=%d",
                req.message[:200],
                0 if req.history is None else len(req.history))

    history_dicts = (
        [m.model_dump() for m in req.history] if req.history is not None else None
    )
    out_messages = run_graph(req.message, history=history_dicts)

    logger.info("chat_response: total_messages=%d", len(out_messages))
    return ChatResponse(messages=[Message(**m) for m in out_messages])


def _as_state(chunk: Any) -> dict:
    """Normalize stream() output to a state dict."""

    if isinstance(chunk, tuple) and len(chunk) >= 1:
        return chunk[0]
    return chunk


def _messages_to_dicts(messages: Iterable[HumanMessage]) -> List[dict]:
    """Convert LangChain messages into simple dicts for the frontend."""

    result: List[dict] = []
    for m in messages:
        role = "assistant"
        if isinstance(m, HumanMessage):
            role = "user"
        result.append({"role": role, "content": m.content})
    return result


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream LangGraph execution steps and final answer."""

    logger.info(
        "chat_stream_request: message=%s, history_len=%d",
        req.message[:200],
        0 if req.history is None else len(req.history),
    )

    history_dicts = (
        [m.model_dump() for m in req.history] if req.history is not None else None
    )

    messages: List[HumanMessage] = []
    history = history_dicts or []
    for item in history:
        if item.get("role") == "user":
            messages.append(HumanMessage(content=item.get("content", "")))
    messages.append(HumanMessage(content=req.message))

    def event_stream() -> Iterable[str]:
        last_node: Optional[str] = None
        final_messages: Optional[List[HumanMessage]] = None

        initial_state = {
            "messages": messages,
            "retrieved_docs": [],
            "search_results": [],
            "route": None,
            "current_node": None,
            "reflect_count": 0,
            "search_query": None,
        }

        for event in graph.stream(
            initial_state, stream_mode=["values", "messages"]
        ):
            if isinstance(event, tuple) and len(event) == 2:
                mode, chunk = event
            else:
                mode, chunk = "values", event

            if mode == "values":
                state = _as_state(chunk)
                node = state.get("current_node")
                if node and node != last_node:
                    evt = {"type": "node", "node": node}
                    yield json.dumps(evt, ensure_ascii=False) + "\n"
                    last_node = node
                state_messages = state.get("messages")
                if state_messages:
                    final_messages = state_messages
                continue

            if mode == "messages":
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    msg_chunk, meta = chunk[0], chunk[1]
                else:
                    msg_chunk, meta = chunk, {}
                node_name = (meta or {}).get("langgraph_node") or ""
                delta = getattr(msg_chunk, "content", "") or ""
                if isinstance(delta, list):
                    delta = "".join(
                        getattr(part, "text", str(part)) for part in delta
                    )
                if delta:
                    evt = {
                        "type": "llm_token",
                        "node": node_name,
                        "delta": delta,
                    }
                    yield json.dumps(evt, ensure_ascii=False) + "\n"

        if final_messages is None:
            final_messages = messages

        dict_messages = _messages_to_dicts(final_messages)
        full_text = ""
        for m in reversed(dict_messages):
            if m["role"] == "assistant":
                full_text = m["content"]
                break

        done_evt = {
            "type": "done",
            "message": full_text,
            "messages": dict_messages,
        }
        yield json.dumps(done_evt, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")

