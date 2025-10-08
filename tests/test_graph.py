"""
Tests for the graph: the nodes, the branch, and the end-to-end run.

The nodes are adapters now — the logic they used to contain lives in
`agents/retrieval.py`, `agents/grounding.py`, `agents/actions.py` and
`agents/approval.py`, each with its own test file. What is left to check here
is the wiring: that state flows the right way, that the branch sends a command
past the vector store, and that a failure in one stage does not take the run
down.
"""

from __future__ import annotations

import pytest

import agents.graph as graph
from agents import approval, retrieval
from tests.fakes import FakeRetriever, default_contexts


@pytest.fixture
def fake_retriever():
    retriever = FakeRetriever()
    retrieval.set_retriever(retriever)
    graph.reset_graph()
    return retriever


@pytest.fixture
def no_llm(monkeypatch):
    """Compose deterministically, without resolving a provider."""
    calls = []

    class Outcome:
        def __init__(self, text):
            self.text = text
            self.provider = "stub"
            self.cost = type("C", (), {"calls": 1, "to_dict": lambda self: {"calls": 1}})()

        def to_dict(self):
            return {"provider": "stub", "cost": {"calls": 1, "total_tokens": 10, "usd": 0.0}}

    def fake_compose(query, contexts, temperature=0.2, style=None, **kwargs):
        calls.append({"query": query, "contexts": contexts, "temperature": temperature, "style": style})
        return Outcome(f"Composed for {query!r}. [finance.pdf p7]")

    monkeypatch.setattr(graph, "compose", fake_compose)
    return calls


class TestRouterNode:
    def test_the_router_reads_its_triggers_from_the_registry(self):
        # Not a literal list in the node: registering an action must teach the
        # router about it without a second edit.
        state = graph.router_node({"query": "pause 'Chicken Biryani'"})
        assert state["intent"] == "action"

    def test_a_question_routes_to_knowledge(self):
        assert graph.router_node({"query": "how are payouts scheduled?"})["intent"] == "knowledge"

    def test_a_missing_query_does_not_raise(self):
        assert graph.router_node({})["intent"] == "knowledge"


class TestRetrieverNode:
    def test_contexts_reach_the_state(self, fake_retriever):
        state = graph.retriever_node({"query": "payouts", "log": []})
        assert len(state["contexts"]) == 2
        assert state["contexts"][0]["metadata"]["filename"] == "finance.pdf"

    def test_latency_is_recorded_for_the_cost_panel(self, fake_retriever):
        state = graph.retriever_node({"query": "payouts", "log": []})
        assert state["retrieval"]["count"] == 2
        assert state["retrieval"]["embed_ms"] >= 0

    def test_top_k_is_clamped_before_it_reaches_retrieval(self, fake_retriever):
        graph.retriever_node({"query": "q", "top_k": 9999, "log": []})
        assert fake_retriever.calls[0]["top_k"] == graph.clamp_top_k(9999)

    @pytest.mark.parametrize("bad", [0, -3, None, "lots"])
    def test_a_nonsense_top_k_still_retrieves_something(self, fake_retriever, bad):
        graph.retriever_node({"query": "q", "top_k": bad, "log": []})
        assert fake_retriever.calls[0]["top_k"] >= 1

    def test_a_failing_index_does_not_take_the_graph_down(self):
        retrieval.set_retriever(FakeRetriever(error="RuntimeError"))
        state = graph.retriever_node({"query": "q", "log": []})
        assert state["contexts"] == []
        assert "retrieval failed" in state["log"][-1]


class TestComposeNode:
    def test_it_passes_the_retrieved_contexts_to_the_composer(self, no_llm):
        contexts = default_contexts()
        graph.compose_node({"query": "q", "contexts": contexts, "log": []})
        assert no_llm[0]["contexts"] == contexts

    def test_style_and_temperature_are_forwarded(self, no_llm):
        graph.compose_node({"query": "q", "contexts": [], "style": "concise", "temperature": 0.7, "log": []})
        assert no_llm[0]["style"] == "concise"
        assert no_llm[0]["temperature"] == 0.7

    def test_the_defaults_are_the_documented_ones(self, no_llm):
        graph.compose_node({"query": "q", "contexts": [], "log": []})
        assert no_llm[0]["style"] == "detailed"
        assert no_llm[0]["temperature"] == 0.2

    def test_the_provider_and_cost_are_recorded_on_the_state(self, no_llm):
        state = graph.compose_node({"query": "q", "contexts": [], "log": []})
        assert state["answer_meta"]["cost"]["calls"] == 1


class TestCriticNode:
    def test_a_valid_citation_counts_as_grounded(self):
        state = graph.critic_node(
            {
                "answer": "Payouts are issued weekly on Tuesdays for the preceding week. [finance.pdf p7]",
                "contexts": default_contexts(),
            }
        )
        assert state["grounded"] is True
        assert "grounded=True" in state["log"][-1]

    def test_prose_without_a_citation_is_not_grounded(self):
        state = graph.critic_node({"answer": "The documents do not cover this.", "contexts": default_contexts()})
        assert state["grounded"] is False

    def test_a_missing_answer_is_not_grounded(self):
        assert graph.critic_node({})["grounded"] is False

    def test_the_report_is_carried_on_the_state_not_only_logged(self):
        state = graph.critic_node({"answer": "x [finance.pdf p7]", "contexts": default_contexts()})
        assert "validity" in state["grounding"]


class TestActionNode:
    def test_a_recognised_command_reaches_the_approval_gate(self):
        state = graph.action_node({"query": "pause 'Chicken Biryani'"})
        assert state["approval_id"]
        request = approval.get_gate().get(state["approval_id"])
        assert request.status is approval.ApprovalStatus.PENDING
        assert request.params["item"] == "Chicken Biryani"

    def test_nothing_is_dispatched_by_the_node(self):
        gate = approval.get_gate()
        graph.action_node({"query": "pause 'Chicken Biryani'"})
        assert gate.dispatcher.dispatched == [], "the action node must have no side effect"

    def test_an_unrecognised_command_is_not_submitted(self):
        state = graph.action_node({"query": "pause '   '"})
        assert state["proposed_action"]["type"] == "unknown"
        assert state["approval_id"] is None
        assert approval.get_gate().pending() == []

    def test_an_unrecognised_command_still_gets_a_user_facing_answer(self):
        assert graph.action_node({"query": "pause '  '"})["answer"]

    def test_the_answer_names_the_risk_and_says_nothing_was_sent(self):
        answer = graph.action_node({"query": "update hours for Saturday"})["answer"]
        assert "high" in answer
        assert "Nothing has been sent" in answer

    def test_a_missing_query_does_not_raise(self):
        assert graph.action_node({})["proposed_action"]["type"] == "unknown"


class TestTopology:
    def test_the_graph_is_described_by_data(self):
        assert [node.name for node in graph.NODES] == [
            "router",
            "retriever",
            "compose",
            "critic",
            "action",
        ]

    def test_both_intents_have_a_destination(self):
        names = {node.name for node in graph.NODES}
        assert set(graph.INTENT_ROUTES.values()) <= names


class TestGraphEndToEnd:
    def test_a_question_walks_router_retriever_compose_critic(self, fake_retriever, no_llm):
        result = graph.invoke_graph("How are payouts scheduled?")
        steps = " | ".join(result["log"])
        assert "Router → Knowledge" in steps
        assert "Retriever → Compose" in steps
        assert "Compose → Critic" in steps
        assert "Critic → END" in steps

    def test_a_question_reaches_the_composer_with_retrieved_text(self, fake_retriever, no_llm):
        graph.invoke_graph("How are payouts scheduled?")
        assert len(no_llm[0]["contexts"]) == 2

    def test_a_command_skips_retrieval_and_the_model_entirely(self, fake_retriever, no_llm):
        result = graph.invoke_graph("pause 'Chicken Biryani'")
        assert fake_retriever.calls == [], "a command must not hit the vector store"
        assert no_llm == [], "a command must not call the model"
        assert result["proposed_action"]["item"] == "Chicken Biryani"
        assert result["approval_id"]

    def test_invoke_graph_forwards_style_and_temperature(self, fake_retriever, no_llm):
        graph.invoke_graph("payouts?", style="concise", temperature=0.5)
        assert no_llm[0]["style"] == "concise"
        assert no_llm[0]["temperature"] == 0.5

    def test_invoke_graph_forwards_top_k(self, fake_retriever, no_llm):
        graph.invoke_graph("payouts?", top_k=7)
        assert fake_retriever.calls[0]["top_k"] == 7

    def test_a_broken_index_still_returns_an_honest_answer(self):
        retrieval.set_retriever(FakeRetriever(error="RuntimeError"))
        graph.reset_graph()
        result = graph.invoke_graph("How are payouts scheduled?")
        assert "no supporting documents" in result["answer"].lower()
        assert result["grounded"] is False

    def test_exactly_one_composition_happens_per_question(self, fake_retriever, no_llm):
        # The headline audit finding: the UI composed once through the graph
        # and then streamed a second, separately billed answer.
        graph.invoke_graph("How are payouts scheduled?")
        assert len(no_llm) == 1
