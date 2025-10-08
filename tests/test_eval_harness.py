"""
Tests for the eval harness.

The harness is a measuring instrument, so what matters is that it reports
honestly: a retrieval miss must not be hidden by a confident answer, and the
unanswerable question must pass by *not* being answered.
"""

from __future__ import annotations

import json

from agents import retrieval
from eval import harness
from tests.fakes import FakeRetriever, default_passages


def setup_retriever():
    retrieval.set_retriever(FakeRetriever(default_passages()))
    from agents import graph

    graph.reset_graph()


class TestQuestionSet:
    def test_the_checked_in_set_loads(self):
        questions = harness.load_questions()
        assert len(questions) >= 5

    def test_every_question_has_an_id_and_a_query(self):
        for question in harness.load_questions():
            assert question["id"] and question["query"]

    def test_the_ids_are_unique(self):
        ids = [q["id"] for q in harness.load_questions()]
        assert len(ids) == len(set(ids))

    def test_the_set_includes_an_unanswerable_question(self):
        # A harness that only asks answerable questions rewards a system that
        # answers everything confidently.
        assert any(q.get("expect_ungrounded") for q in harness.load_questions())


class TestRunQuestion:
    def test_a_retrieval_miss_is_reported_as_a_miss(self):
        setup_retriever()
        result = harness.run_question(
            {"id": "x", "query": "payouts?", "expect_sources": ["nowhere.pdf"]}
        )
        assert result.retrieval_hit is False
        assert result.passed is False

    def test_a_retrieval_hit_is_reported(self):
        setup_retriever()
        result = harness.run_question(
            {"id": "x", "query": "payouts?", "expect_sources": ["finance.pdf"]}
        )
        assert result.retrieval_hit is True

    def test_the_unanswerable_question_passes_by_not_being_answered(self):
        retrieval.set_retriever(FakeRetriever(error="RuntimeError"))
        from agents import graph

        graph.reset_graph()
        result = harness.run_question(
            {"id": "u", "query": "parental leave in Portugal?", "expect_ungrounded": True}
        )
        assert result.grounded is False
        assert result.passed is True

    def test_expected_phrases_are_counted(self):
        setup_retriever()
        result = harness.run_question(
            {"id": "x", "query": "payouts?", "expect_sources": [], "expect_phrases": ["Payouts", "zzz"]}
        )
        assert result.phrases_expected == 2
        assert result.phrases_found == 1

    def test_latency_and_cost_are_recorded(self):
        setup_retriever()
        result = harness.run_question({"id": "x", "query": "payouts?"})
        assert result.latency_ms >= 0
        assert result.turn_cost.usd == 0.0, "the keyless path must cost nothing"


class TestReport:
    def test_the_report_summarises_a_run(self):
        setup_retriever()
        report = harness.run(
            [
                {"id": "a", "query": "payouts?", "expect_sources": ["finance.pdf"]},
                {"id": "b", "query": "leave?", "expect_ungrounded": True},
            ]
        )
        assert len(report.results) == 2
        assert 0.0 <= report.retrieval_recall <= 1.0

    def test_the_unanswerable_question_is_excluded_from_recall(self):
        setup_retriever()
        report = harness.run([{"id": "b", "query": "leave?", "expect_ungrounded": True}])
        assert report.retrieval_recall == 0.0

    def test_the_report_is_json_serialisable(self):
        setup_retriever()
        report = harness.run([{"id": "a", "query": "payouts?", "expect_sources": ["finance.pdf"]}])
        assert json.loads(json.dumps(report.to_dict()))["questions"] == 1

    def test_the_table_renders(self):
        setup_retriever()
        report = harness.run([{"id": "a", "query": "payouts?", "expect_sources": ["finance.pdf"]}])
        table = harness.format_table(report)
        assert "passed" in table and "cost" in table

    def test_an_empty_run_does_not_divide_by_zero(self):
        report = harness.run([])
        assert report.retrieval_recall == 0.0
        assert report.median_latency_ms == 0.0
