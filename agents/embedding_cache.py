"""
A content-addressed cache for embedding vectors.

Embedding is the dominant cost in this system, in both senses. Indexing the
sample corpus spends almost all of its wall-clock time inside
``SentenceTransformer.encode``, and every query pays for one more encode before
Chroma is touched at all. Neither of those costs is inherent: the same text
embedded with the same model always produces the same vector.

Two places benefit, for different reasons:

* **Indexing.** Re-running ``build_index`` after adding one PDF re-embeds the
  entire corpus, because the indexer has no memory of what it has already seen.
  With the cache, a rebuild costs one encode per *new or changed* chunk. That
  is the difference between a minute and a second on a corpus this size, and it
  scales with the corpus rather than with the edit.
* **Querying.** Operations questions repeat — the same handful of "how do
  payouts work" phrasings arrive all day. A repeat query skips the model
  entirely and goes straight to Chroma.

The key includes the model name, so switching ``EMBEDDING_MODEL`` cannot serve
vectors from the previous model. The store is SQLite because it needs to
survive a restart, be safe across the indexer and the UI running at once, and
add no service to the compose file; ``NullEmbeddingCache`` disables the whole
thing when ``EMBEDDING_CACHE_PATH`` is empty.
"""

from __future__ import annotations

import array
import hashlib
import logging
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from agents import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key    TEXT PRIMARY KEY,
    dim    INTEGER NOT NULL,
    vector BLOB NOT NULL
);
"""


def cache_key(model: str, text: str) -> str:
    """Stable identity for (model, text).

    The model name is part of the digest rather than a separate column so that
    a stale row for a different model is simply a miss, not a wrong hit.
    """
    digest = hashlib.blake2b(digest_size=20)
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache(Protocol):
    """Somewhere to keep vectors between runs."""

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, list[float]]:
        """Return ``{text: vector}`` for the texts that are present."""

    def put_many(self, model: str, pairs: Sequence[tuple[str, list[float]]]) -> None:
        """Store vectors for later."""


class NullEmbeddingCache:
    """Caches nothing. The honest way to express "disabled"."""

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, list[float]]:
        return {}

    def put_many(self, model: str, pairs: Sequence[tuple[str, list[float]]]) -> None:
        return None


class SqliteEmbeddingCache:
    """Vectors on disk, keyed by content digest.

    Vectors are stored as a packed little-endian float32 array rather than JSON:
    a 384-dimension vector is 1.5KB packed against roughly 8KB as text, and the
    corpus has thousands of them.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One connection guarded by a lock. The alternative — a connection per
        # call — reopens the file thousands of times during an index build.
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        # The indexer and the UI can hold the file at the same time; WAL lets
        # the reader carry on while the writer commits.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, list[float]]:
        if not texts:
            return {}
        keys = {cache_key(model, text): text for text in texts}
        found: dict[str, list[float]] = {}
        # Chunked so a large batch does not exceed SQLite's variable limit.
        key_list = list(keys)
        with self._lock:
            for start in range(0, len(key_list), 400):
                window = key_list[start : start + 400]
                placeholders = ",".join("?" * len(window))
                rows = self._connection.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                    window,
                ).fetchall()
                for key, blob in rows:
                    vector = array.array("f")
                    vector.frombytes(blob)
                    found[keys[key]] = list(vector)
        return found

    def put_many(self, model: str, pairs: Sequence[tuple[str, list[float]]]) -> None:
        if not pairs:
            return
        rows = [
            (cache_key(model, text), len(vector), array.array("f", vector).tobytes())
            for text, vector in pairs
        ]
        with self._lock:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                rows,
            )
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


_cache: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    """The process-wide cache, built from configuration on first use."""
    global _cache
    if _cache is None:
        _cache = build_embedding_cache(config.EMBEDDING_CACHE_PATH)
    return _cache


def build_embedding_cache(path: str) -> EmbeddingCache:
    """A cache for ``path``, degrading to no cache if it cannot be opened.

    A cache is an optimisation. An unwritable directory — a read-only mount, a
    volume owned by another user — must slow the system down, not stop it.
    """
    if not path:
        return NullEmbeddingCache()
    try:
        return SqliteEmbeddingCache(path)
    except (sqlite3.Error, OSError):
        logger.warning("Embedding cache at %s is unusable; running without it.", path, exc_info=True)
        return NullEmbeddingCache()


def set_embedding_cache(cache: EmbeddingCache | None) -> None:
    """Install a cache (or ``None`` to rebuild from config). Used by tests."""
    global _cache
    _cache = cache


def embed_cached(embedder, texts: Sequence[str], cache: EmbeddingCache | None = None) -> list[list[float]]:
    """Embed ``texts``, encoding only the ones that are not already cached.

    Order is preserved, and duplicates within a single batch are encoded once —
    which matters more than it sounds: chunk overlap means neighbouring chunks
    of a repetitive document are often byte-identical.
    """
    texts = list(texts)
    if not texts:
        return []
    cache = cache if cache is not None else get_embedding_cache()
    model_name = getattr(embedder, "model_name", None) or getattr(embedder, "name", "unknown")

    hits = cache.get_many(model_name, texts)
    missing = [text for text in dict.fromkeys(texts) if text not in hits]
    if missing:
        vectors = embedder.encode(missing)
        fresh = list(zip(missing, vectors, strict=False))
        cache.put_many(model_name, fresh)
        hits.update(dict(fresh))
    return [hits[text] for text in texts]
