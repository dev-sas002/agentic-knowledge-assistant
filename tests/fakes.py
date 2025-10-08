"""
Stand-ins for the hosted model, the embedder and the vector store.

Nothing here talks to a network. `FakeGroq` records the arguments it was
constructed and called with, so the tests can assert on the request that
*would* have been sent — the timeout, the retry count, the model, and the
shape of the prompt — without a key, a bill, or a flaky dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

from agents.retrieval import Passage, RetrievalResult


def _message(content: str, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _delta(content: str, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
        usage=usage,
    )


class FakeCompletions:
    def __init__(self, owner: FakeGroq) -> None:
        self._owner = owner

    def create(self, **kwargs: Any):
        self._owner.calls.append(kwargs)
        if self._owner.raise_on_create is not None:
            raise self._owner.raise_on_create
        if kwargs.get("stream"):
            return _stream(self._owner.stream_chunks, self._owner.raise_mid_stream, self._owner.usage)
        return _message(self._owner.content, self._owner.usage)


def _stream(chunks: Iterable[str], raise_after: Exception | None, usage: Any):
    for chunk in chunks:
        yield _delta(chunk)
    if raise_after is not None:
        raise raise_after
    if usage is not None:
        yield _delta("", usage=usage)


class FakeGroq:
    """Mimics the slice of the Groq SDK that `agents.providers` actually uses."""

    #: Every instance ever constructed, with its constructor kwargs.
    instances: ClassVar[list[FakeGroq]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = kwargs
        self.calls: list[dict[str, Any]] = []
        self.content: str = "A generated answer. [finance.pdf p7]"
        self.stream_chunks: list[str] = ["A ", "generated ", "answer."]
        self.raise_on_create: Exception | None = None
        self.raise_mid_stream: Exception | None = None
        self.usage: Any = None
        self.chat = SimpleNamespace(completions=FakeCompletions(self))
        FakeGroq.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    @classmethod
    def latest(cls) -> FakeGroq:
        assert cls.instances, "no Groq client was constructed"
        return cls.instances[-1]


class FakeEmbedder:
    """Satisfies the ``Embedder`` protocol; records what it was asked to encode."""

    name = "fake"
    model_name = "fake-model"
    dimension = 3

    def __init__(self, vector: list[float] | None = None) -> None:
        self.encoded: list[list[str]] = []
        self.vector = vector or [0.1, 0.2, 0.3]

    def encode(self, texts: Sequence[str], normalize: bool = True) -> list[list[float]]:
        self.encoded.append(list(texts))
        return [list(self.vector) for _ in texts]


class FakeCollection:
    """A Chroma collection whose query result is whatever the test supplies."""

    def __init__(self, result: dict | None = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else default_chroma_result()
        self.error = error
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict:
        self.queries.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeRetriever:
    """Returns fixed passages. Used where the test is about what happens *after* retrieval."""

    def __init__(self, passages: list[Passage] | None = None, error: str | None = None) -> None:
        self.passages = passages if passages is not None else default_passages()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        self.calls.append({"query": query, "top_k": top_k})
        if self.error:
            return RetrievalResult(passages=[], embed_ms=0.1, query_ms=0.0, error=self.error)
        return RetrievalResult(passages=list(self.passages[:top_k]), embed_ms=1.0, query_ms=2.0)


def default_chroma_result() -> dict:
    return {
        "documents": [
            [
                "Payouts are issued weekly on Tuesdays for the preceding week. "
                "A payout is held if the bank account fails verification.",
                "Finance reconciles disputed charges within five business days.",
            ]
        ],
        "metadatas": [
            [
                {"filename": "finance.pdf", "page": 7, "title": "Finance"},
                {"filename": "finance.pdf", "page": 8, "title": "Finance"},
            ]
        ],
        "distances": [[0.12, 0.31]],
    }


def default_passages() -> list[Passage]:
    raw = default_chroma_result()
    return [
        Passage(document=doc, metadata=meta, distance=dist)
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0], strict=False)
    ]


def default_contexts() -> list[dict]:
    return [passage.to_dict() for passage in default_passages()]
