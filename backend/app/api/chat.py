from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.graph import run_graph


router = APIRouter(prefix="/api", tags=["chat"])


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

    history_dicts = (
        [m.model_dump() for m in req.history] if req.history is not None else None
    )
    out_messages = run_graph(req.message, history=history_dicts)
    return ChatResponse(messages=[Message(**m) for m in out_messages])

