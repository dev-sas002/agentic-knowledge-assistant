"""
Tests for the critic.

The old check was one regex: a citation anywhere meant grounded. These tests
are mostly the answers that check passed and should not have.
"""

from __future__ import annotations

from agents.grounding import assess, citations, context_sources

FINANCE = {
    "document": (
        "Payouts are issued weekly on Tuesdays for the preceding week. A payout "
        "is held if the bank account fails verification."
    ),
    "metadata": {"filename": "finance.pdf", "page": 7},
}
MENU = {
    "document": "Items can be paused for stock-outs. To unpause, verify availability.",
    "metadata": {"filename": "menu.pdf", "page": 2},
}
CONTEXTS = [FINANCE, MENU]


class TestCitationExtraction:
    def test_citations_are_found(self):
        assert citations("Payouts run weekly. [finance.pdf p7]") == ["finance.pdf p7"]

    def test_a_bracketed_non_citation_is_not_a_citation(self):
        assert citations("See [the handbook] for details.") == []

    def test_citations_are_deduplicated_but_ordered(self):
        found = citations("a [menu.pdf p2] b [finance.pdf p7] c [menu.pdf p2]")
        assert found == ["menu.pdf p2", "finance.pdf p7"]

    def test_matching_is_case_insensitive(self):
        assert citations("[Finance.PDF p7]") == citations("[finance.pdf p7]")

    def test_an_unknown_page_marker_is_still_a_citation(self):
        assert citations("[finance.pdf p?]") == ["finance.pdf p?"]


class TestValidity:
    def test_a_fabricated_citation_is_caught(self):
        # The single most important check here: a plausible filename that was
        # never retrieved used to pass as grounded.
        report = assess("Leave is 20 days. [handbook_2019.pdf p3]", CONTEXTS)
        assert report.grounded is False
        assert report.unknown_citations == ["handbook_2019.pdf p3"]
        assert report.validity == 0.0

    def test_the_reason_names_the_fabricated_source(self):
        report = assess("Leave is 20 days. [handbook_2019.pdf p3]", CONTEXTS)
        assert "handbook_2019.pdf p3" in report.reason

    def test_a_mix_of_real_and_fabricated_citations_scores_partially(self):
        report = assess(
            "Payouts are issued weekly on Tuesdays. [finance.pdf p7] "
            "Leave is twenty days per year for all staff. [handbook.pdf p1]",
            CONTEXTS,
        )
        assert 0.0 < report.validity < 1.0


class TestCoverage:
    def test_a_single_citation_does_not_ground_nine_invented_sentences(self):
        answer = " ".join(
            [f"This is an entirely invented claim about operational policy number {n}." for n in range(9)]
        ) + " Payouts are issued weekly on Tuesdays. [finance.pdf p7]"
        report = assess(answer, CONTEXTS)
        assert report.coverage < 0.3
        assert report.grounded is False

    def test_a_fully_cited_answer_scores_well(self):
        answer = (
            "Payouts are issued weekly on Tuesdays for the preceding week. [finance.pdf p7]\n"
            "A payout is held if the bank account fails verification. [finance.pdf p7]"
        )
        report = assess(answer, CONTEXTS)
        assert report.coverage == 1.0
        assert report.grounded is True

    def test_short_scaffolding_lines_are_not_held_against_the_answer(self):
        answer = (
            "Summary:\n"
            "Payouts are issued weekly on Tuesdays for the preceding week. [finance.pdf p7]"
        )
        assert assess(answer, CONTEXTS).coverage == 1.0

    def test_uncited_sentences_are_named_so_the_operator_can_judge(self):
        answer = (
            "Payouts are issued weekly on Tuesdays for the preceding week. [finance.pdf p7] "
            "Refunds are processed within twenty four hours of the request being raised."
        )
        assert assess(answer, CONTEXTS).uncited_sentences


class TestSupport:
    def test_a_citation_to_an_unrelated_passage_scores_low_on_support(self):
        report = assess("Items can be paused for stock-outs during service. [finance.pdf p7]", CONTEXTS)
        assert report.support < 0.5

    def test_a_citation_to_the_right_passage_scores_high_on_support(self):
        report = assess("Payouts are issued weekly on Tuesdays for the preceding week. [finance.pdf p7]", CONTEXTS)
        assert report.support > 0.8


class TestEdges:
    def test_no_citation_at_all_is_not_grounded(self):
        report = assess("The documents do not cover this.", CONTEXTS)
        assert report.grounded is False
        assert report.score == 0.0
        assert "cites no sources" in report.reason

    def test_an_empty_answer_is_not_grounded(self):
        assert assess("", CONTEXTS).grounded is False

    def test_an_answer_with_no_retrieved_context_cannot_be_grounded(self):
        assert assess("Payouts run weekly. [finance.pdf p7]", []).grounded is False

    def test_context_sources_normalises_case_and_missing_pages(self):
        sources = context_sources([{"metadata": {"filename": "A.PDF"}}])
        assert sources == {"a.pdf p?"}

    def test_a_source_field_is_used_when_filename_is_missing(self):
        assert context_sources([{"metadata": {"source": "data/raw/a.pdf", "page": 1}}]) == {"data/raw/a.pdf p1"}

    def test_the_report_serialises_everything_the_ui_renders(self):
        data = assess("Payouts run weekly. [finance.pdf p7]", CONTEXTS).to_state()
        for field in ("grounded", "score", "coverage", "validity", "support", "reason"):
            assert field in data


class TestAgreementWithTheComposer:
    def test_the_composer_emits_the_citation_format_the_critic_reads(self):
        from agents.local_composer import _citation

        assert citations(_citation({"filename": "a.pdf", "page": 4})) == ["a.pdf p4"]

    def test_the_composers_non_answer_is_correctly_reported_as_ungrounded(self):
        from agents.local_composer import compose_extractive

        text = compose_extractive("parental leave in Portugal", CONTEXTS)
        if "[" not in text:
            assert assess(text, CONTEXTS).grounded is False
