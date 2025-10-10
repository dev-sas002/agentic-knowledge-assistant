"""
Retrieval: question in, ranked passages out.

Split out of the graph node so that "what did we retrieve" can be tested,
measured and swapped without constructing a LangGraph at all. The node in
``agents/graph.py`` is now four lines that call this.

Everything downstream — prompting, extractive composition, the grounding
check, the UI's source panel — consumes the dicts this returns. That shape is
therefore part of the contract and is documented on ``Passage``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agents import config
from agents.embedding_cache import embed_cached
from agents.providers import Embedder, resolve_embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk.

    ``distance`` is Chroma's, where smaller is closer; ``similarity`` is the
    same number turned the right way up for a human, because a progress bar
    that fills as relevance *falls* is a UI bug waiting to happen.
    """

    document: str
    metadata: dict[str, Any]
    distance: float | None

    @property
    def similarity(self) -> float | None:
        if self.distance is None:
            return None
        # Vectors are L2-normalised at both index and query time, so Chroma's
        # squared-L2 distance lies in [0, 4] and maps linearly onto [1, -1].
        return max(0.0, min(1.0, 1.0 - (self.distance / 2.0)))

    @property
    def source(self) -> str:
        meta = self.metadata or {}
        filename = meta.get("filename") or meta.get("source") or "unknown"
        return f"{filename} p{meta.get('page', '?')}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "metadata": self.metadata,
            "distance": self.distance,
            "similarity": self.similarity,
            "source": self.source,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """What was retrieved, and what it cost to retrieve it."""

    passages: list[Passage]
    embed_ms: float
    query_ms: float
    error: str | None = None

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.query_ms

    def as_contexts(self) -> list[dict[str, Any]]:
        return [passage.to_dict() for passage in self.passages]


class Retriever(Protocol):
    """Anything that can turn a question into ranked passages."""

    def retrieve(self, query: str, top_k: int) -> RetrievalResult: ...


def clamp_top_k(value: Any, maximum: int | None = None) -> int:
    """Bound the fan-out that arrives from the UI.

    ``top_k`` is the one retrieval parameter a user controls, and it multiplies
    directly into prompt size. Clamping here rather than trusting the widget
    means the API and the eval harness get the same guarantee.
    """
    maximum = config.MAX_TOP_K if maximum is None else maximum
    try:
        wanted = int(value)
    except (TypeError, ValueError):
        wanted = 4
    return max(1, min(wanted, maximum))


def _first_row(result: dict, key: str) -> list:
    """Chroma returns ``{key: [[...]]}``, or None when a field was not included."""
    rows = result.get(key) or []
    if not rows:
        return []
    return rows[0] or []


class ChromaRetriever:
    """The real retriever: a local embedding model over a persistent Chroma collection."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        collection: Any | None = None,
        use_cache: bool = True,
    ) -> None:
        self._embedder = embedder
        self._collection = collection
        self._use_cache = use_cache

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = resolve_embedder()
        return self._embedder

    @property
    def collection(self) -> Any:
        if self._collection is None:
            # Imported lazily: chromadb is a heavy dependency, and merely
            # importing the routing and action logic must not require it.
            import chromadb

            client = chromadb.PersistentClient(path=str(Path(config.CHROMA_DB_DIR)))
            self._collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
        return self._collection

    def embed_query(self, query: str) -> list[float]:
        if self._use_cache:
            return embed_cached(self.embedder, [query])[0]
        return list(self.embedder.encode([query])[0])

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        top_k = clamp_top_k(top_k)
        started = time.perf_counter()
        try:
            vector = self.embed_query(query)
            embedded = time.perf_counter()
            raw = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["distances", "metadatas", "documents"],
            )
            finished = time.perf_counter()
        except Exception as exc:
            # An unreachable or unbuilt index must not take the graph down:
            # the composer already has an honest "nothing retrieved" answer.
            logger.exception("Retrieval failed")
            elapsed = (time.perf_counter() - started) * 1000
            return RetrievalResult(passages=[], embed_ms=elapsed, query_ms=0.0, error=type(exc).__name__)

        documents = _first_row(raw, "documents")
        metadatas = _first_row(raw, "metadatas")
        distances = _first_row(raw, "distances")
        # `zip` over three rows that Chroma may return at different lengths
        # would silently truncate; pad instead so a missing distance shows as
        # "n/a" rather than dropping the passage entirely.
        passages = [
            Passage(
                document=documents[i],
                # Chroma can return a null metadata entry; normalise it so every
                # consumer can treat `.metadata` as a dict.
                metadata=(metadatas[i] if i < len(metadatas) else None) or {},
                distance=distances[i] if i < len(distances) else None,
            )
            for i in range(len(documents))
        ]
        return RetrievalResult(
            passages=passages,
            embed_ms=(embedded - started) * 1000,
            query_ms=(finished - embedded) * 1000,
        )


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """The process-wide retriever."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever


def set_retriever(retriever: Retriever | None) -> None:
    """Install a retriever (or ``None`` to rebuild). Used by tests and the eval harness."""
    global _retriever
    _retriever = retriever
