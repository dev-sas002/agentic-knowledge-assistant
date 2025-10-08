"""
Tests for routing and action parsing.

The action path proposes operational changes for a human to approve. The worst
failure it can have is naming the wrong item confidently: the approval screen
looks correct and the system acts on something else. Most of these tests exist
because that is exactly what it did.
"""

from __future__ import annotations

import pytest

from agents.graph import action_node, router_node


def route(query: str) -> str:
    return router_node({"query": query})["intent"]


def propose(query: str) -> dict:
    return action_node({"query": query})["proposed_action"]


class TestRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "How are merchant payouts scheduled?",
            "What is the onboarding process?",
            "Explain the reconciliation policy",
        ],
    )
    def test_questions_route_to_knowledge(self, query):
        assert route(query) == "knowledge"

    @pytest.mark.parametrize(
        "query",
        [
            "pause 'Chicken Biryani'",
            "unpause the lamb dish",
            "update hours for Saturday",
        ],
    )
    def test_commands_route_to_action(self, query):
        assert route(query) == "action"


class TestActionItemExtraction:
    """
    The regression. The original pattern's non-greedy group stopped at the
    first word boundary, so "pause 'Chicken Biryani'" proposed pausing
    "Chicken".
    """

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("pause 'Chicken Biryani'", "Chicken Biryani"),
            ('unpause "Lamb Karahi Special"', "Lamb Karahi Special"),
            ("pause Chicken Biryani", "Chicken Biryani"),
            ("resume 'Fish and Chips'", "Fish and Chips"),
            ("pause 'Soup'", "Soup"),
        ],
    )
    def test_the_whole_item_name_is_captured(self, query, expected):
        assert propose(query)["item"] == expected

    def test_pause_and_unpause_map_to_different_actions(self):
        assert propose("pause 'X'")["type"] == "pause_item"
        assert propose("unpause 'X'")["type"] == "unpause_item"

    def test_resume_is_treated_as_unpause(self):
        assert propose("resume 'X'")["type"] == "unpause_item"

    def test_the_original_query_is_retained_for_the_approval_screen(self):
        # A reviewer approving an action should be able to see what was typed,
        # not only what the parser made of it.
        query = "pause 'Chicken Biryani'"
        assert propose(query)["original_query"] == query


class TestHoursAction:
    def test_an_hours_change_is_recognised(self):
        assert propose("update hours to 9am-11pm on weekends")["type"] == "update_hours"

    def test_the_details_are_kept(self):
        proposed = propose("set hours to 10:00-22:00")
        assert "10:00-22:00" in proposed["details"]


class TestUnrecognisedActions:
    def test_an_unparseable_command_is_reported_as_unknown(self):
        # Better to propose nothing than to guess at an operational change.
        assert propose("do the thing with the stuff")["type"] == "unknown"

    def test_an_unknown_action_carries_no_item(self):
        assert "item" not in propose("do the thing with the stuff")
