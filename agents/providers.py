"""
Provider seams for the two things this system buys from somewhere else: the
answer model and the embedding model.

Both are registries keyed by name, resolved from the environment. Adding a
second hosted model — OpenAI, Bedrock, a self-hosted vLLM endpoint — is a new
class and one ``@register_answer_provider`` line; nothing in the graph, the
UI or the tests changes. That is the seam a future developer on this project
actually needs, because "which model" is the decision most likely to be
revisited and the one most entangled with cost.

The registry is also what makes the keyless mode honest rather than a special
case: ``ExtractiveAnswerProvider`` is a first-class provider implementing the
same protocol, not an ``if api_key is None`` branch buried in the call site.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import struct
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agents import config
from agents.local_composer import compose_extractive
from agents.prompting import build_prompt, system_message

logger = logging.getLogger(__name__)

# Optional dependency: only needed when a Groq key is configured.
try:  # pragma: no cover - exercised by substitution in tests
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore


# --------------------------------------------------------------------------
# Answer providers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AnswerRequest:
    """Everything a provider needs to compose one answer.

    Deliberately not a rendered prompt: the extractive provider does not use a
    prompt at all, and a future provider may want to use a native
    citations/grounding API rather than free text. Each provider renders what
    it needs from these fields.
    """

    query: str
    contexts: list[dict] = field(default_factory=list)
    style: str | None = None
    temperature: float = 0.2
    max_tokens: int = config.MAX_ANSWER_TOKENS


@runtime_checkable
class AnswerProvider(Protocol):
    """A source of answers over retrieved context."""

    name: str

    def available(self) -> bool:
        """True when this provider is configured and usable right now."""

    def complete(self, request: AnswerRequest) -> str:
        """Return a complete answer."""

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        """Yield the answer progressively, in order."""

    def last_usage(self) -> Any | None:
        """Token usage reported for this thread's most recent call, if any.

        Optional on purpose: the extractive provider has no usage to report,
        and a future provider may not return one. Callers fall back to
        ``agents.cost``'s estimate when this is ``None``.
        """


_ANSWER_PROVIDERS: dict[str, Callable[[], AnswerProvider]] = {}
_ANSWER_CACHE: dict[str, AnswerProvider] = {}


def register_answer_provider(name: str) -> Callable[[Callable[[], AnswerProvider]], Callable[[], AnswerProvider]]:
    def decorator(factory: Callable[[], AnswerProvider]) -> Callable[[], AnswerProvider]:
        _ANSWER_PROVIDERS[name] = factory
        return factory

    return decorator


def answer_provider_names() -> list[str]:
    return sorted(_ANSWER_PROVIDERS)


class ExtractiveAnswerProvider:
    """Composes an answer from sentences that are verbatim in the sources.

    Costs nothing, needs no key, and cannot hallucinate — which is why it is
    also the fallback when a hosted provider fails mid-turn.
    """

    name = "extractive"

    def available(self) -> bool:
        return True

    def complete(self, request: AnswerRequest) -> str:
        return compose_extractive(request.query, request.contexts, style=request.style)

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        # Streamed line by line so the UI's progressive-reveal path is
        # exercised offline too, rather than being code that only runs when
        # someone is paying for tokens.
        yield from self.complete(request).splitlines(True)

    def last_usage(self) -> Any | None:
        return None


class GroqAnswerProvider:
    """The hosted path: Groq's OpenAI-compatible chat completions."""

    name = "groq"

    def __init__(self) -> None:
        self._client: Any | None = None
        # Thread-local: Streamlit serves each session on its own thread, and a
        # shared attribute would report another user's token count.
        self._local = threading.local()

    def last_usage(self) -> Any | None:
        return getattr(self._local, "usage", None)

    # -- construction --------------------------------------------------
    def client(self) -> Any | None:
        """The SDK client, or None when it cannot be built.

        The key is read at call time, not at import time: this module is
        imported before Streamlit or compose have necessarily wired the
        environment.
        """
        if self._client is not None:
            return self._client
        api_key = config.env_str("GROQ_API_KEY", config.GROQ_API_KEY)
        if not api_key or Groq is None:
            return None
        try:
            self._client = Groq(
                api_key=api_key,
                timeout=config.GROQ_TIMEOUT_SECONDS,
                max_retries=config.GROQ_MAX_RETRIES,
            )
        except Exception:  # pragma: no cover - construction rarely fails
            # Degrade to another provider rather than taking the app down.
            logger.exception("Could not construct the Groq client")
            self._client = None
        return self._client

    def available(self) -> bool:
        return self.client() is not None

    # -- completion ----------------------------------------------------
    def _messages(self, request: AnswerRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_message(request.style)},
            {"role": "user", "content": build_prompt(request.query, request.contexts)},
        ]

    def complete(self, request: AnswerRequest) -> str:
        client = self.client()
        if client is None:
            raise ProviderUnavailable("Groq is not configured")
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=self._messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        self._local.usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return (getattr(choices[0].message, "content", "") or "").strip()

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        client = self.client()
        if client is None:
            raise ProviderUnavailable("Groq is not configured")
        stream = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=self._messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        self._local.usage = None
        for chunk in stream:
            # Groq attaches usage to the final chunk of a stream.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._local.usage = usage
            try:
                delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                yield delta


class ProviderUnavailable(RuntimeError):
    """Raised when a provider is asked to work without being configured."""


register_answer_provider("extractive")(ExtractiveAnswerProvider)
register_answer_provider("groq")(GroqAnswerProvider)

#: Tried in order when ``LLM_PROVIDER=auto``. First one that reports itself
#: available wins; ``extractive`` always does, so resolution never fails.
AUTO_ORDER = ("groq", "extractive")


def get_answer_provider(name: str) -> AnswerProvider:
    """Build (and cache) the named provider."""
    if name not in _ANSWER_PROVIDERS:
        raise KeyError(f"Unknown answer provider {name!r}. Known: {answer_provider_names()}")
    if name not in _ANSWER_CACHE:
        _ANSWER_CACHE[name] = _ANSWER_PROVIDERS[name]()
    return _ANSWER_CACHE[name]


def resolve_answer_provider(name: str | None = None) -> AnswerProvider:
    """The provider to use for this turn.

    ``auto`` (the default) walks ``AUTO_ORDER`` and takes the first available
    one, so the system runs with or without a key and the difference is a
    configuration fact rather than a code path.
    """
    requested = (name or config.env_str("LLM_PROVIDER", config.LLM_PROVIDER)).lower()
    if requested != "auto":
        provider = get_answer_provider(requested)
        if provider.available():
            return provider
        logger.warning("Provider %r is not available; falling back.", requested)
    for candidate in AUTO_ORDER:
        provider = get_answer_provider(candidate)
        if provider.available():
            return provider
    return get_answer_provider("extractive")


def reset_providers() -> None:
    """Drop cached provider instances. Used by tests and after a config change."""
    _ANSWER_CACHE.clear()
    _EMBEDDER_CACHE.clear()


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. The indexer and the retriever share one."""

    name: str
    dimension: int

    def encode(self, texts: Sequence[str], normalize: bool = True) -> list[list[float]]:
        """Return one vector per input text."""


_EMBEDDERS: dict[str, Callable[[], Embedder]] = {}
_EMBEDDER_CACHE: dict[str, Embedder] = {}


def register_embedder(name: str) -> Callable[[Callable[[], Embedder]], Callable[[], Embedder]]:
    def decorator(factory: Callable[[], Embedder]) -> Callable[[], Embedder]:
        _EMBEDDERS[name] = factory
        return factory

    return decorator


def embedder_names() -> list[str]:
    return sorted(_EMBEDDERS)


class SentenceTransformerEmbedder:
    """The real one: a local bge-small model, no network call per query."""

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model: Any | None = None
        self._dimension = 384

    @property
    def model(self) -> Any:
        if self._model is None:
            # Imported lazily: this pulls torch, and merely importing the
            # routing and action logic must not require the model stack.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Older model cards do not expose the dimension; the default stands.
            with contextlib.suppress(Exception):  # pragma: no cover
                self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], normalize: bool = True) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=normalize)
        return vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]


class HashingEmbedder:
    """A deterministic bag-of-words hash embedder with no dependencies.

    Not semantic, and it says so: two paraphrases that share no vocabulary get
    unrelated vectors. It exists so the pipeline can be indexed and queried in
    CI, or on a laptop with no 130MB download and no torch, and so the
    embedding seam has a second implementation proving it is a seam rather
    than a wrapper around one library.
    """

    name = "hashing"

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], normalize: bool = True) -> list[list[float]]:
        return [self._one(text, normalize) for text in texts]

    def _one(self, text: str, normalize: bool) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _words(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = struct.unpack("<Q", digest)[0] % self._dimension
            vector[bucket] += 1.0
        if normalize:
            length = math.sqrt(sum(v * v for v in vector))
            if length:
                vector = [v / length for v in vector]
        return vector


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


register_embedder("sentence-transformers")(SentenceTransformerEmbedder)
register_embedder("hashing")(HashingEmbedder)


def resolve_embedder(name: str | None = None) -> Embedder:
    """Build (and cache) the configured embedder."""
    requested = (name or config.env_str("EMBEDDER_PROVIDER", config.EMBEDDER_PROVIDER)).lower()
    if requested not in _EMBEDDERS:
        raise KeyError(f"Unknown embedder {requested!r}. Known: {embedder_names()}")
    if requested not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[requested] = _EMBEDDERS[requested]()
    return _EMBEDDER_CACHE[requested]
