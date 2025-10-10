"""
One place that reads the environment.

Every other module imports its settings from here rather than calling
``os.environ`` inline. Two reasons, both of which bit this project before:

* a malformed value used to crash at *import* time — ``int(os.environ["MAX_TOP_K"])``
  with ``MAX_TOP_K=lots`` took down the whole Streamlit process with a
  ``ValueError`` and no indication of which variable was at fault;
* the defaults were scattered across four modules, so the README's
  configuration table had four places to drift out of step with.

The helpers below fall back to the documented default and log, rather than
raising: a bad environment variable should degrade one setting, not the app.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _coerced(name: str, default: T, cast: Callable[[str], T]) -> T:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring %s=%r: not a valid %s. Using the default %r.",
            name,
            raw,
            cast.__name__,
            default,
        )
        return default


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    return _coerced(name, default, int)


def env_float(name: str, default: float) -> float:
    return _coerced(name, default, float)


# ---- Retrieval and indexing ----
CHROMA_DB_DIR = env_str("CHROMA_DB_DIR", "data/chroma")
RAW_DOCS_DIR = env_str("RAW_DOCS_DIR", "data/raw")
CHROMA_COLLECTION = env_str("CHROMA_COLLECTION", "pwb_docs")
EMBEDDING_MODEL = env_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDER_PROVIDER = env_str("EMBEDDER_PROVIDER", "sentence-transformers")
#: Upper bound on retrieval fan-out. ``top_k`` arrives from the UI, and an
#: unbounded value would push an arbitrary amount of text into the prompt.
MAX_TOP_K = env_int("MAX_TOP_K", 12)
#: Where the embedding cache lives. Empty disables it.
EMBEDDING_CACHE_PATH = env_str("EMBEDDING_CACHE_PATH", "data/embedding_cache.sqlite3")

# ---- Answer provider ----
#: ``auto`` picks Groq when a key is present and the extractive composer when
#: it is not. Naming a provider explicitly overrides that.
LLM_PROVIDER = env_str("LLM_PROVIDER", "auto")
GROQ_API_KEY = env_str("GROQ_API_KEY", "")
GROQ_MODEL = env_str("GROQ_MODEL", "openai/gpt-oss-20b")
#: A hosted call with no timeout hangs the Streamlit worker indefinitely; a
#: transient 5xx with no retry loses the turn.
GROQ_TIMEOUT_SECONDS = env_float("GROQ_TIMEOUT_SECONDS", 30.0)
GROQ_MAX_RETRIES = env_int("GROQ_MAX_RETRIES", 2)
MAX_ANSWER_TOKENS = env_int("MAX_ANSWER_TOKENS", 768)

# ---- Prompt trust boundary ----
#: Hard cap on retrieved document text placed inside the ``<context>`` fence.
#: Independent of ``top_k``: a single pathological chunk must not be able to
#: fill the window on its own.
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 12000)

# ---- Human-approval gate ----
OPS_API_URL = env_str("OPS_API_URL", "http://localhost:8001")
OPS_DATA_DIR = env_str("OPS_DATA_DIR", "data")
OPS_REQUEST_TIMEOUT = env_float("OPS_REQUEST_TIMEOUT", 10.0)
OPERATOR_NAME = env_str("OPERATOR_NAME", "demo_user")
#: How long a proposed action stays approvable. An approval screen left open
#: overnight must not still be able to change tomorrow's menu.
APPROVAL_TTL_SECONDS = env_float("APPROVAL_TTL_SECONDS", 900.0)
