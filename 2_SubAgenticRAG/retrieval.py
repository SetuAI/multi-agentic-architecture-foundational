"""
retrieval.py
============

The RAG plumbing, kept separate from the agent logic.

Its job is small and self-contained:
  1. read the four quarterly transcripts from data/,
  2. split them into chunks,
  3. embed the chunks with a local Ollama embedding model,
  4. hand back a `retrieve(query, k)` function the sub-agents can call.

Each sub-agent in subagentic_rag.py calls `retrieve` with its OWN query, scoped
to its specialty. That is what makes this a *sub-agentic RAG* rather than a plain
RAG: retrieval happens inside each specialist, not once up front.

The embeddings object is passed in (dependency injection) so the same code can run
with real Ollama embeddings in production and a fake one in offline tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"

# Embedding model name; override with OLLAMA_EMBED_MODEL. nomic-embed-text is the
# reliable local default for RAG.
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embeddings():
    """Return a real Ollama embeddings client. Imported lazily so the module can be
    imported (and tested with a fake) even on a machine without Ollama installed."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL)


def _load_chunks() -> list[Document]:
    """Read every transcript and split it into overlapping chunks.

    The `source` metadata (e.g. 'nimbus_q3_fy26') travels with each chunk so that a
    sub-agent's finding can say WHICH quarter a fact came from."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path.stem}))
    return docs


def build_vectorstore(embeddings=None) -> InMemoryVectorStore:
    """Build an in-memory vector store from the transcripts.

    In-memory keeps the demo setup-free (no database files). For a persistent store
    you would swap InMemoryVectorStore for Chroma here and change nothing else."""
    embeddings = embeddings or get_embeddings()
    store = InMemoryVectorStore(embeddings)
    store.add_documents(_load_chunks())
    return store


def make_retriever(vectorstore: InMemoryVectorStore) -> Callable[[str, int], list[tuple[str, str]]]:
    """Return a simple retrieve(query, k) function.

    Each call returns a list of (chunk_text, source_label) pairs — the evidence a
    sub-agent reasons over."""
    def retrieve(query: str, k: int = 4) -> list[tuple[str, str]]:
        hits = vectorstore.similarity_search(query, k=k)
        return [(d.page_content, d.metadata.get("source", "unknown")) for d in hits]
    return retrieve