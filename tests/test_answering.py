"""
Tests for the answer policy.

The question these answer is "what does the operator see when the provider has
a bad day?". The answer must always be an answer — never an error string, never
two answers stitched together, and never a second billed call.
"""

from __future__ import annotations

import pytest

from agents import answering, providers
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

QUERY = "how are payouts scheduled?"


@pytest.fixture
def groq(monkeypatch):
    FakeGroq.reset()
    monkeypatch.setattr(providers, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    providers.reset_providers()
    assert providers.resolve_answer_provider().name == "groq"
    return FakeGroq


class TestOneCallPerQuestion:
    def test_composing_makes_exactly_one_request(self, groq):
        answering.compose(QUERY, [FINANCE])
        assert len(FakeGroq.latest().calls) == 1

    def test_the_outcome_carries_the_text_so_the_caller_need_not_ask_again(self, groq):
        outcome = answering.compose(QUERY, [FINANCE])
        assert outcome.text
        assert outcome.cost.calls == 1

    def test_the_wrapper_and_compose_agree(self, groq):
        assert answering.answer_with_citations(QUERY, [FINANCE]) == FakeGroq.latest().content


class TestKeylessOperation:
    def test_without_a_key_an_answer_is_still_produced(self):
        outcome = answering.compose(QUERY, [FINANCE])
        assert outcome.provider == "extractive"
        assert "finance.pdf p7" in outcome.text

    def test_the_keyless_path_costs_nothing_and_says_so(self):
        assert answering.compose(QUERY, [FINANCE]).cost.usd == 0.0

    def test_no_context_produces_an_honest_non_answer(self):
        text = answering.compose(QUERY, []).text
        assert "no supporting documents" in text.lower()

    def test_the_non_answer_carries_no_citation(self):
        # A non-answer must fail the grounding check, not pass it.
        from agents.grounding import citations

        assert citations(answering.compose(QUERY, []).text) == []


class TestFallback:
    def test_a_failing_provider_still_answers_from_the_retrieved_text(self, groq):
        FakeGroq.latest().raise_on_create = RuntimeError("503 service unavailable")
        outcome = answering.compose(QUERY, [FINANCE])
        assert outcome.fell_back is True
        assert "finance.pdf p7" in outcome.text

    def test_the_fallback_answer_is_not_an_error_string(self, groq):
        FakeGroq.latest().raise_on_create = RuntimeError("503")
        text = answering.compose(QUERY, [FINANCE]).text
        assert "503" not in text and "Traceback" not in text

    def test_an_empty_completion_falls_back(self, groq):
        FakeGroq.latest().content = "   "
        assert answering.compose(QUERY, [FINANCE]).fell_back is True

    def test_the_fallback_names_what_failed(self, groq):
        FakeGroq.latest().raise_on_create = RuntimeError("503")
        assert "groq" in answering.compose(QUERY, [FINANCE]).provider


class TestStreaming:
    def test_fragments_accumulate_into_the_answer(self, groq):
        assert "".join(answering.stream_answer_with_citations(QUERY, [FINANCE])) == "A generated answer."

    def test_a_stream_that_dies_before_emitting_falls_back(self, groq):
        FakeGroq.latest().raise_on_create = RuntimeError("connection reset")
        text = "".join(answering.stream_answer_with_citations(QUERY, [FINANCE]))
        assert "finance.pdf p7" in text

    def test_a_stream_that_dies_after_emitting_is_not_restarted(self, groq):
        # Appending a second answer to a half-rendered one produces two
        # answers stitched together on screen.
        FakeGroq.latest().raise_mid_stream = RuntimeError("connection reset")
        text = "".join(answering.stream_answer_with_citations(QUERY, [FINANCE]))
        assert text.startswith("A generated answer.")
        assert text.endswith(answering.STREAM_CUT_SHORT)

    def test_a_stream_that_yields_nothing_falls_back(self, groq):
        FakeGroq.latest().stream_chunks = []
        assert "finance.pdf p7" in "".join(answering.stream_answer_with_citations(QUERY, [FINANCE]))

    def test_the_keyless_path_streams_too(self):
        # The progressive-reveal path must be exercised offline, not only when
        # someone is paying for tokens.
        assert "".join(answering.stream_answer_with_citations(QUERY, [FINANCE]))


class TestCostReporting:
    def test_reported_usage_is_preferred_over_the_estimate(self, groq):
        FakeGroq.latest().usage = {"prompt_tokens": 1234, "completion_tokens": 321}
        outcome = answering.compose(QUERY, [FINANCE])
        assert outcome.cost.input_tokens == 1234
        assert outcome.cost.output_tokens == 321

    def test_without_reported_usage_the_cost_is_estimated(self, groq):
        FakeGroq.latest().usage = None
        assert answering.compose(QUERY, [FINANCE]).cost.input_tokens > 0

    def test_more_context_costs_more(self, groq):
        one = answering.compose(QUERY, [FINANCE]).cost.input_tokens
        many = answering.compose(QUERY, [FINANCE] * 4).cost.input_tokens
        assert many > one
