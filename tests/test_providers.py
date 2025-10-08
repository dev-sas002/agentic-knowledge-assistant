"""
Tests for the provider seam.

Every one of these runs against `tests.fakes.FakeGroq`; the `no_network`
fixture in conftest turns a real socket into an immediate failure, so a
regression that starts calling the hosted API shows up as a failing test
rather than as a bill.

The properties under test are the ones that decide whether a turn survives a
bad day at the provider: a request that cannot hang, a request that is
retried, and a registry that resolves to something usable with or without a key.
"""

from __future__ import annotations

import pytest

from agents import config, providers
from tests.fakes import FakeGroq

FINANCE = {
    "document": (
        "Payouts are issued weekly on Tuesdays for the preceding week. A payout "
        "is held if the bank account fails verification. Finance reconciles "
        "disputed charges within five business days."
    ),
    "metadata": {"filename": "finance.pdf", "page": 7},
    "distance": 0.1,
}


def request(query: str = "how are payouts scheduled?", contexts=None) -> providers.AnswerRequest:
    return providers.AnswerRequest(query=query, contexts=contexts if contexts is not None else [FINANCE])


@pytest.fixture
def groq(monkeypatch):
    """Install FakeGroq as the SDK and return the configured provider."""
    FakeGroq.reset()
    monkeypatch.setattr(providers, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    providers.reset_providers()
    provider = providers.get_answer_provider("groq")
    assert provider.available()
    return provider


class TestRegistry:
    def test_both_providers_are_registered(self):
        assert providers.answer_provider_names() == ["extractive", "groq"]

    def test_an_unknown_provider_is_a_clear_error(self):
        with pytest.raises(KeyError) as excinfo:
            providers.get_answer_provider("gpt-9")
        assert "extractive" in str(excinfo.value), "the error should name what is available"

    def test_providers_are_cached(self):
        assert providers.get_answer_provider("extractive") is providers.get_answer_provider("extractive")

    def test_registering_a_provider_is_one_line(self):
        @providers.register_answer_provider("stub-for-test")
        class Stub:
            name = "stub-for-test"

            def available(self):
                return True

            def complete(self, req):
                return "stub"

            def stream(self, req):
                yield "stub"

        try:
            assert providers.get_answer_provider("stub-for-test").complete(request()) == "stub"
        finally:
            providers._ANSWER_PROVIDERS.pop("stub-for-test", None)
            providers.reset_providers()


class TestResolution:
    def test_without_a_key_the_extractive_provider_is_chosen(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        providers.reset_providers()
        assert providers.resolve_answer_provider().name == "extractive"

    def test_with_a_key_groq_is_chosen(self, groq):
        assert providers.resolve_answer_provider().name == "groq"

    def test_an_explicit_provider_overrides_auto(self, groq):
        assert providers.resolve_answer_provider("extractive").name == "extractive"

    def test_an_unavailable_explicit_provider_falls_back_rather_than_failing(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        providers.reset_providers()
        assert providers.resolve_answer_provider("groq").name == "extractive"

    def test_resolution_never_fails(self, monkeypatch):
        # `extractive` is always available, so there is no "no provider" state
        # for a caller to handle.
        monkeypatch.setattr(providers, "Groq", None)
        providers.reset_providers()
        assert providers.resolve_answer_provider() is not None


class TestGroqConstruction:
    def test_no_key_means_no_client(self, monkeypatch):
        monkeypatch.setattr(providers, "Groq", FakeGroq)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        providers.reset_providers()
        assert providers.get_answer_provider("groq").available() is False

    def test_the_sdk_being_absent_means_no_client(self, monkeypatch):
        # `groq` is an optional dependency; its absence must degrade, not crash.
        monkeypatch.setattr(providers, "Groq", None)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        providers.reset_providers()
        assert providers.get_answer_provider("groq").available() is False

    def test_the_key_is_read_at_call_time_not_import_time(self, monkeypatch):
        # The module is imported before the environment is necessarily wired.
        FakeGroq.reset()
        monkeypatch.setattr(providers, "Groq", FakeGroq)
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        providers.reset_providers()
        assert providers.get_answer_provider("groq").available() is False
        monkeypatch.setenv("GROQ_API_KEY", "arrived-late")
        providers.reset_providers()
        assert providers.get_answer_provider("groq").available() is True

    def test_a_timeout_is_always_set(self, groq):
        # A hosted call with no timeout hangs the Streamlit worker forever.
        assert FakeGroq.latest().init_kwargs["timeout"] == config.GROQ_TIMEOUT_SECONDS

    def test_retries_are_configured(self, groq):
        assert FakeGroq.latest().init_kwargs["max_retries"] == config.GROQ_MAX_RETRIES

    def test_a_client_that_will_not_construct_degrades_instead_of_crashing(self, monkeypatch):
        class Exploding:
            def __init__(self, **kwargs):
                raise RuntimeError("bad config")

        monkeypatch.setattr(providers, "Groq", Exploding)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        providers.reset_providers()
        assert providers.get_answer_provider("groq").available() is False


class TestGroqCompletion:
    def test_the_configured_model_is_used(self, groq):
        groq.complete(request())
        assert FakeGroq.latest().calls[0]["model"] == config.GROQ_MODEL

    def test_the_prompt_carries_the_retrieved_text_inside_the_fence(self, groq):
        groq.complete(request())
        user_message = FakeGroq.latest().calls[0]["messages"][1]["content"]
        assert "<context>" in user_message
        assert "Payouts are issued weekly" in user_message

    def test_an_unconfigured_provider_raises_rather_than_returning_nonsense(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        providers.reset_providers()
        with pytest.raises(providers.ProviderUnavailable):
            providers.get_answer_provider("groq").complete(request())

    def test_an_empty_choices_list_yields_an_empty_string(self, groq):
        FakeGroq.latest().content = ""
        assert groq.complete(request()) == ""

    def test_usage_is_reported_when_the_provider_gives_it(self, groq):
        FakeGroq.latest().usage = {"prompt_tokens": 1200, "completion_tokens": 300}
        groq.complete(request())
        assert groq.last_usage()["prompt_tokens"] == 1200


class TestGroqStreaming:
    def test_fragments_arrive_in_order(self, groq):
        assert "".join(groq.stream(request())) == "A generated answer."

    def test_streaming_is_requested_of_the_sdk(self, groq):
        list(groq.stream(request()))
        assert FakeGroq.latest().calls[0]["stream"] is True

    def test_a_chunk_with_no_delta_is_skipped_not_fatal(self, groq):
        FakeGroq.latest().stream_chunks = ["A", "", None, "B"]
        assert "".join(groq.stream(request())) == "AB"


class TestExtractiveProvider:
    def test_it_is_always_available(self):
        assert providers.get_answer_provider("extractive").available() is True

    def test_it_answers_from_the_retrieved_text_with_a_citation(self):
        answer = providers.get_answer_provider("extractive").complete(request())
        assert "finance.pdf p7" in answer

    def test_it_streams_the_same_text_it_completes(self):
        provider = providers.get_answer_provider("extractive")
        assert "".join(provider.stream(request())) == provider.complete(request())

    def test_it_reports_no_usage(self):
        assert providers.get_answer_provider("extractive").last_usage() is None


class TestEmbedders:
    def test_both_embedders_are_registered(self):
        assert providers.embedder_names() == ["hashing", "sentence-transformers"]

    def test_the_hashing_embedder_needs_no_dependencies(self):
        vectors = providers.HashingEmbedder(dimension=64).encode(["payouts are weekly"])
        assert len(vectors) == 1 and len(vectors[0]) == 64

    def test_the_hashing_embedder_is_deterministic(self):
        embedder = providers.HashingEmbedder(dimension=64)
        assert embedder.encode(["payouts"]) == embedder.encode(["payouts"])

    def test_normalised_vectors_have_unit_length(self):
        vector = providers.HashingEmbedder(dimension=64).encode(["payouts are weekly"])[0]
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-6

    def test_empty_text_does_not_divide_by_zero(self):
        assert providers.HashingEmbedder(dimension=8).encode([""]) == [[0.0] * 8]

    def test_an_unknown_embedder_is_a_clear_error(self):
        with pytest.raises(KeyError):
            providers.resolve_embedder("word2vec")

    def test_the_configured_embedder_is_resolved_and_cached(self, monkeypatch):
        monkeypatch.setenv("EMBEDDER_PROVIDER", "hashing")
        providers.reset_providers()
        assert providers.resolve_embedder() is providers.resolve_embedder()
        assert providers.resolve_embedder().name == "hashing"
