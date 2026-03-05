from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # DeepSeek
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    deepseek_model: str = Field("deepseek-chat", env="DEEPSEEK_MODEL")

    # Zhipu Web Search
    # 字段名与 .env 中的 ZHIPUAI_API_KEY 对应
    zhipuai_api_key: str = Field(...)

    # Vector store / RAG
    # 若设置 CHROMA_HOST，则连接 Docker 中的 Chroma 服务；否则使用本地目录
    chroma_host: Optional[str] = Field(None, env="CHROMA_HOST")
    chroma_port: int = Field(8001, env="CHROMA_PORT")
    chroma_collection: str = Field("knowledge_base", env="CHROMA_COLLECTION")
    chroma_persist_dir: Path = Field(
        Path("./chroma_db"), env="CHROMA_PERSIST_DIR"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


settings = get_settings()

