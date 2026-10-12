"""RAG sobre diretrizes clinicas: ingestao, vector store e retriever."""

from src.rag.ingestion import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_text,
    ingest_document,
    load_pdf,
    load_text,
)
from src.rag.retriever import Retriever
from src.rag.types import Chunk, RetrievalResult
from src.rag.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDER,
    VectorStore,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDER",
    "DEFAULT_OVERLAP",
    "Chunk",
    "RetrievalResult",
    "Retriever",
    "VectorStore",
    "chunk_text",
    "ingest_document",
    "load_pdf",
    "load_text",
]
