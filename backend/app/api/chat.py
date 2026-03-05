from typing import List, Optional
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.graph import run_graph


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

