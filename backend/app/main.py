import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router


def configure_logging() -> None:
    """Configure root logger for simple console output."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Knowledge Base GPT Chat Demo")

    # Allow local frontends (e.g., http://localhost:5173 or file://) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)

    @app.get("/")
    async def root() -> dict:
        return {"message": "Knowledge Base GPT Chat Backend is running."}

    return app


app = create_app()

