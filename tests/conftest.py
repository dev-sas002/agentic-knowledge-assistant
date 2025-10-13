"""
Shared test setup.

`tests/` has no `__init__.py`, so pytest puts the *test* directory on
`sys.path` rather than the project root: `pytest tests/` failed to import
`agents` at all, and the suite only ran via `python -m pytest`, which happens
to add the working directory. Putting the root on the path here makes both
invocations work.

Three autouse fixtures enforce the rules this suite is built on, and each one
exists because the alternative is a test that quietly does the wrong thing:

* `no_network` turns any outbound socket into a loud failure, so no test can
  start spending tokens or depending on a live service;
* `clean_llm_env` strips a developer's real key out of the environment and
  drops the cached provider, so a machine with `GROQ_API_KEY` set does not
  take a different code path from CI;
* `isolated_gate` gives every test its own approval gate with a dispatcher
  that records instead of executing — which is what makes "does a rejected
  action send anything?" a question a unit test can answer.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Turn any outbound socket into an immediate, obvious failure."""

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "A test attempted a real network connection. Mock the client instead."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """No test inherits a developer's real key, or another test's provider."""
    from agents import config, providers

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto", raising=False)
    providers.reset_providers()
    yield
    providers.reset_providers()


@pytest.fixture(autouse=True)
def isolated_gate():
    """A fresh approval gate per test, wired to a dispatcher that executes nothing."""
    from agents import approval

    dispatcher = approval.RecordingDispatcher()
    approval.set_gate(approval.ApprovalGate(dispatcher=dispatcher, ttl_seconds=900.0))
    yield approval.get_gate()
    approval.set_gate(None)


@pytest.fixture(autouse=True)
def no_embedding_cache():
    """Tests must not read or write a developer's on-disk vector cache."""
    from agents import embedding_cache

    embedding_cache.set_embedding_cache(embedding_cache.NullEmbeddingCache())
    yield
    embedding_cache.set_embedding_cache(None)


@pytest.fixture(autouse=True)
def reset_retriever():
    from agents import retrieval

    retrieval.set_retriever(None)
    yield
    retrieval.set_retriever(None)
