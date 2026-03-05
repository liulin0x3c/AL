from pathlib import Path
from typing import Iterable, List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.embeddings import get_embeddings


def get_vectorstore(persist_directory: Path | None = None) -> VectorStore:
    """Create or load a Chroma vector store. 支持本地目录或 Docker 中的 HTTP 服务。"""

    embeddings = get_embeddings()

    if settings.chroma_host:
        # 连接 Docker 中的 Chroma 服务
        host = settings.chroma_host
        port = settings.chroma_port
        client = chromadb.HttpClient(host=host, port=port)
        return Chroma(
            client=client,
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
        )
    else:
        # 本地持久化模式
        persist_dir = Path(persist_directory or settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split raw documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return list(splitter.split_documents(list(docs)))

