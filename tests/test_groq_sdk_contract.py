"""
The Groq provider against the real SDK, with the HTTP layer replaced.

``tests/fakes.FakeGroq`` stands in for the SDK everywhere else, which proves
what the provider asks for but not that the installed ``groq`` package still
accepts it and still returns objects of the shape the provider reads. These
tests keep the real client and swap only its transport for an in-process
``httpx.MockTransport``, so an SDK upgrade that renames an argument or moves
a field fails here, without a key and without a socket.
"""

from __future__ import annotations

import json

import httpx
import pytest

groq = pytest.importorskip("groq")

from agents import config, providers  # noqa: E402

FINANCE = {
    "document": "Payouts are issued weekly on Tuesdays for the preceding week.",
    "metadata": {"filename": "finance.pdf", "page": 7},
    "distance": 0.1,
}


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": config.GROQ_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129},
    }


def _chunk(content: str, usage: dict | None = None) -> str:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": config.GROQ_MODEL,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    if usage is not None:
        body["x_groq"] = {"id": "req-test", "usage": usage}
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


@pytest.fixture
def sent():
    return []


@pytest.fixture
def provider(monkeypatch, sent):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append({"url": str(request.url), "payload": payload})
        if payload.get("stream"):
            stream = _chunk("Tuesdays ") + _chunk("[finance.pdf p7]", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}) + "data: [DONE]\n\n"
            return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion("Payouts go out on Tuesdays. [finance.pdf p7]"))

    def real_client(**kwargs):
        return groq.Groq(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(providers, "Groq", real_client)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    providers.reset_providers()
    return providers.get_answer_provider("groq")


def test_the_sdk_accepts_the_client_options(provider):
    client = provider.client()
    assert isinstance(client, groq.Groq)
    assert client.max_retries == config.GROQ_MAX_RETRIES


def test_a_completion_round_trips_through_the_sdk(provider, sent):
    request = providers.AnswerRequest(query="When are payouts issued?", contexts=[FINANCE])
    assert provider.complete(request) == "Payouts go out on Tuesdays. [finance.pdf p7]"
    assert sent[0]["url"].endswith("/openai/v1/chat/completions")
    assert sent[0]["payload"]["model"] == config.GROQ_MODEL
    assert provider.last_usage().total_tokens == 129


def test_a_stream_round_trips_through_the_sdk(provider):
    request = providers.AnswerRequest(query="When are payouts issued?", contexts=[FINANCE])
    assert "".join(provider.stream(request)) == "Tuesdays [finance.pdf p7]"
