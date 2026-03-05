"""
Simple ingestion script to build the local knowledge base.

It scans the ../data directory for text-like files, splits them into chunks,
and writes them into a persistent Chroma vector store.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.rag.vectorstore import chunk_documents, get_vectorstore


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents() -> List[Document]:
    docs: List[Document] = []
    if not DATA_DIR.exists():
        return docs

    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def main() -> None:
    print(f"Loading documents from: {DATA_DIR}")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    if not docs:
        print("No documents found. Put some .txt or .md files into the data/ directory.")
        return

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vs = get_vectorstore()
    vs.add_documents(chunks)
    print("Ingestion completed.")


if __name__ == "__main__":
    main()

