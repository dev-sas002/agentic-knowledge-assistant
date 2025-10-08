"""
Tests for token and cost accounting.

The numbers are estimates, so these tests pin relationships rather than
magnitudes: more context costs more, two calls cost twice one, and the keyless
path costs nothing.
"""

from __future__ import annotations

from agents import cost

FINANCE = {
    "document": "Payouts are issued weekly on Tuesdays. " * 20,
    "metadata": {"filename": "finance.pdf", "page": 7},
}


class TestEstimateTokens:
    def test_empty_text_is_zero(self):
        assert cost.estimate_tokens("") == 0

    def test_longer_text_costs_more(self):
        assert cost.estimate_tokens("x" * 400) > cost.estimate_tokens("x" * 100)

    def test_any_text_at_all_costs_at_least_one(self):
        assert cost.estimate_tokens("x") == 1


class TestEstimateTurn:
    def test_more_context_costs_more(self):
        one = cost.estimate_turn("q", [FINANCE])
        many = cost.estimate_turn("q", [FINANCE] * 4)
        assert many.input_tokens > one.input_tokens

    def test_two_calls_cost_twice_one(self):
        # The audit finding, as a number: the UI made two calls per question.
        one = cost.estimate_turn("q", [FINANCE], answer="a", calls=1)
        two = cost.estimate_turn("q", [FINANCE], answer="a", calls=2)
        assert two.input_tokens == 2 * one.input_tokens
        assert two.usd == 2 * one.usd

    def test_the_extractive_path_costs_nothing(self):
        turn = cost.estimate_turn("q", [FINANCE], provider="extractive")
        assert turn.usd == 0.0 and turn.calls == 0

    def test_output_tokens_are_assumed_when_the_answer_is_unknown(self):
        assert cost.estimate_turn("q", [FINANCE]).output_tokens > 0


class TestFromUsage:
    def test_a_usage_object_is_read(self):
        class Usage:
            prompt_tokens = 1200
            completion_tokens = 300

        turn = cost.from_usage(Usage())
        assert turn.input_tokens == 1200 and turn.output_tokens == 300

    def test_a_usage_dict_is_read(self):
        turn = cost.from_usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert turn.total_tokens == 15

    def test_no_usage_yields_none_so_the_caller_can_estimate(self):
        assert cost.from_usage(None) is None

    def test_a_usage_object_without_token_counts_yields_none(self):
        assert cost.from_usage({"something_else": 1}) is None


class TestAggregation:
    def test_totals_add_up(self):
        turns = [cost.TurnCost(1, 100, 50), cost.TurnCost(1, 200, 60)]
        assert cost.total(turns).input_tokens == 300
        assert cost.total(turns).calls == 2

    def test_an_empty_run_totals_to_zero(self):
        assert cost.total([]).usd == 0.0


class TestBreakdown:
    def test_retrieved_context_dominates(self):
        rows = {row["part"]: row["tokens"] for row in cost.breakdown("q", [FINANCE] * 4)}
        assert rows["retrieved context"] > rows["system prompt"]

    def test_no_context_means_no_context_tokens(self):
        rows = {row["part"]: row["tokens"] for row in cost.breakdown("q", [])}
        assert rows["retrieved context"] == 0
