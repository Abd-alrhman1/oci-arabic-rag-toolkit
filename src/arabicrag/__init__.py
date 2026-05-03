"""arabicrag — production Arabic RAG on Oracle Cloud Infrastructure."""

from .chunking import Chunk, chunk_documents, chunk_by_sections, chunk_by_sentences
from .embeddings import (
    BM25Retriever,
    Embedder,
    HuggingFaceEmbedder,
    OCIEmbedder,
    TfidfEmbedder,
)
from .normalization import normalize
from .prompts import build_messages, format_passages
from .retrieval import HybridRetriever, RetrievalResult, VectorIndex

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Embedder",
    "HuggingFaceEmbedder",
    "HybridRetriever",
    "OCIEmbedder",
    "RetrievalResult",
    "TfidfEmbedder",
    "VectorIndex",
    "build_messages",
    "chunk_by_sections",
    "chunk_by_sentences",
    "chunk_documents",
    "format_passages",
    "normalize",
]
