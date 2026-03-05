from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


@lru_cache()
def get_embeddings() -> Embeddings:
    """Return a multilingual embedding model suitable for Chinese knowledge bases."""

    # sentence-transformers model with good Chinese support
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

