"""
Tests for the human-approval gate.

This is the module that stands between a parsed command and a real change to
the business, so the tests are written as the questions an incident review
would ask. Every one of them is a way the gate could be wrong while still
looking right on screen.
"""

from __future__ import annotations

import threading

import pytest

from agents import approval
from agents.actions import ProposedAction, Risk
from agents.approval import (
    ApprovalError,
    ApprovalGate,
    ApprovalStatus,
    InMemoryApprovalStore,
    OpsApiDispatcher,
    RecordingDispatcher,
    UnknownAction,
)


class Clock:
    """A clock the test moves by hand, so TTL behaviour needs no sleeping."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def pause(item: str = "Chicken Biryani") -> ProposedAction:
    return ProposedAction(type="pause_item", params={"item": item}, original_query=f"pause '{item}'")


@pytest.fixture
def gate():
    return ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=900.0)


class TestSubmission:
    def test_a_submitted_action_is_pending_and_nothing_is_dispatched(self, gate):
        request = gate.submit(pause())
        assert request.status is ApprovalStatus.PENDING
        assert gate.dispatcher.dispatched == []

    def test_the_proposal_carries_what_the_operator_typed(self, gate):
        # A reviewer approving an action must be able to see what was asked
        # for, not only what the parser made of it.
        request = gate.submit(pause())
        assert request.original_query == "pause 'Chicken Biryani'"
        assert "Chicken Biryani" in request.summary

    def test_the_risk_comes_from_the_registry(self, gate):
        assert gate.submit(pause()).risk is Risk.MEDIUM

    def test_an_unregistered_action_cannot_even_be_proposed(self, gate):
        with pytest.raises(UnknownAction):
            gate.submit(ProposedAction(type="delete_everything", params={}, original_query="x"))

    def test_an_unknown_proposal_cannot_be_proposed(self, gate):
        with pytest.raises(UnknownAction):
            gate.submit(ProposedAction(type="unknown", original_query="do the thing"))

    def test_each_submission_gets_its_own_id(self, gate):
        assert gate.submit(pause("A")).id != gate.submit(pause("B")).id


class TestApproval:
    def test_approval_dispatches_exactly_once(self, gate):
        request = gate.submit(pause())
        gate.approve(request.id, decided_by="alice")
        assert len(gate.dispatcher.dispatched) == 1
        assert gate.get(request.id).status is ApprovalStatus.EXECUTED

    def test_the_endpoint_comes_from_the_registry_not_the_request(self, gate):
        request = gate.submit(pause())
        gate.approve(request.id)
        assert gate.dispatcher.dispatched[0]["endpoint"] == "/pause_item"

    def test_a_double_click_cannot_dispatch_twice(self, gate):
        # The naïve version transitioned state after dispatching, so a second
        # click sent a second request.
        request = gate.submit(pause())
        gate.approve(request.id)
        with pytest.raises(ApprovalError):
            gate.approve(request.id)
        assert len(gate.dispatcher.dispatched) == 1

    def test_concurrent_approvals_dispatch_once(self, gate):
        request = gate.submit(pause())
        errors = []

        def attempt():
            try:
                gate.approve(request.id)
            except ApprovalError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(gate.dispatcher.dispatched) == 1
        assert len(errors) == 7

    def test_the_decision_maker_is_recorded(self, gate):
        request = gate.submit(pause())
        gate.approve(request.id, decided_by="alice")
        assert gate.get(request.id).decided_by == "alice"

    def test_the_approval_id_travels_with_the_payload(self, gate):
        # This is what joins the ops audit log to the decision that authorised
        # the change; without it the log says what happened but not who agreed.
        request = gate.submit(pause())
        gate.approve(request.id)
        assert gate.dispatcher.dispatched[0]["payload"]["approval_id"] == request.id

    def test_approving_an_unknown_id_is_an_error_not_a_silent_no_op(self, gate):
        with pytest.raises(ApprovalError):
            gate.approve("does-not-exist")


class TestRejection:
    def test_rejection_sends_nothing(self, gate):
        request = gate.submit(pause())
        gate.reject(request.id, decided_by="bob", reason="wrong dish")
        assert gate.dispatcher.dispatched == []
        assert gate.get(request.id).status is ApprovalStatus.REJECTED

    def test_the_reason_is_kept(self, gate):
        request = gate.submit(pause())
        gate.reject(request.id, reason="wrong dish")
        assert gate.get(request.id).reason == "wrong dish"

    def test_a_rejected_action_cannot_later_be_approved(self, gate):
        request = gate.submit(pause())
        gate.reject(request.id)
        with pytest.raises(ApprovalError):
            gate.approve(request.id)
        assert gate.dispatcher.dispatched == []

    def test_rejection_is_recorded_in_the_history(self, gate):
        request = gate.submit(pause())
        gate.reject(request.id, decided_by="bob")
        events = [entry["event"] for entry in gate.get(request.id).history]
        assert events == ["proposed", "rejected"]


class TestExpiry:
    def test_a_stale_proposal_expires(self):
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        request = gate.submit(pause())
        clock.advance(61)
        assert gate.get(request.id).status is ApprovalStatus.EXPIRED

    def test_an_expired_proposal_cannot_be_approved(self):
        # The screen may still show the button; the gate must still refuse.
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        request = gate.submit(pause())
        clock.advance(61)
        with pytest.raises(ApprovalError):
            gate.approve(request.id)
        assert gate.dispatcher.dispatched == []

    def test_expiry_is_rechecked_at_approval_time_not_only_at_render_time(self):
        # A request can expire between the page being drawn and the click.
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        request = gate.submit(pause())
        assert gate.get(request.id).status is ApprovalStatus.PENDING  # rendered
        clock.advance(61)  # the operator went to lunch
        with pytest.raises(ApprovalError):
            gate.approve(request.id)

    def test_a_proposal_inside_its_ttl_still_works(self):
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        request = gate.submit(pause())
        clock.advance(59)
        gate.approve(request.id)
        assert len(gate.dispatcher.dispatched) == 1

    def test_expired_proposals_drop_out_of_pending(self):
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        gate.submit(pause())
        assert len(gate.pending()) == 1
        clock.advance(61)
        assert gate.pending() == []

    def test_seconds_remaining_never_goes_negative(self):
        clock = Clock()
        gate = ApprovalGate(dispatcher=RecordingDispatcher(), ttl_seconds=60.0, clock=clock)
        request = gate.submit(pause())
        clock.advance(600)
        assert request.seconds_remaining(clock()) == 0.0


class TestDispatchFailure:
    def test_a_failing_ops_api_is_recorded_not_swallowed(self):
        class Broken:
            def dispatch(self, request, spec):
                raise ConnectionError("ops api unreachable")

        gate = ApprovalGate(dispatcher=Broken(), ttl_seconds=60.0)
        request = gate.submit(pause())
        result = gate.approve(request.id)
        assert result.status is ApprovalStatus.FAILED
        assert "ConnectionError" in result.error

    def test_a_failed_dispatch_is_distinct_from_a_rejection(self):
        # "The human said no" and "the human said yes and it broke" need
        # different follow-up, so they are different states.
        class Broken:
            def dispatch(self, request, spec):
                raise ConnectionError("boom")

        gate = ApprovalGate(dispatcher=Broken(), ttl_seconds=60.0)
        request = gate.submit(pause())
        assert gate.approve(request.id).status is not ApprovalStatus.REJECTED

    def test_a_failure_is_terminal_and_not_silently_retried(self):
        class Broken:
            def dispatch(self, request, spec):
                raise ConnectionError("boom")

        gate = ApprovalGate(dispatcher=Broken(), ttl_seconds=60.0)
        request = gate.submit(pause())
        gate.approve(request.id)
        with pytest.raises(ApprovalError):
            gate.approve(request.id)


class TestOpsApiDispatcher:
    def test_the_url_is_built_from_the_registry_endpoint(self, monkeypatch):
        sent = {}

        class Response:
            ok = True

            def json(self):
                return {"ok": True}

        def fake_post(url, json=None, timeout=None):
            sent.update({"url": url, "json": json, "timeout": timeout})
            return Response()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        gate = ApprovalGate(dispatcher=OpsApiDispatcher(base_url="http://ops:8001"), ttl_seconds=60.0)
        request = gate.submit(pause())
        gate.approve(request.id)
        assert sent["url"] == "http://ops:8001/pause_item"
        assert sent["json"]["item"] == "Chicken Biryani"

    def test_a_non_ok_response_fails_the_request(self, monkeypatch):
        class Response:
            ok = False
            status_code = 400
            text = "Missing 'item' in payload"

        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: Response())
        gate = ApprovalGate(dispatcher=OpsApiDispatcher(base_url="http://ops:8001"), ttl_seconds=60.0)
        request = gate.submit(pause())
        assert gate.approve(request.id).status is ApprovalStatus.FAILED


class TestStore:
    def test_terminal_requests_are_evicted_before_pending_ones(self):
        store = InMemoryApprovalStore(capacity=3)
        gate = ApprovalGate(store=store, dispatcher=RecordingDispatcher(), ttl_seconds=900.0)
        kept = gate.submit(pause("keep me"))
        for index in range(6):
            decided = gate.submit(pause(f"item-{index}"))
            gate.reject(decided.id)
        assert gate.get(kept.id) is not None

    def test_the_module_level_gate_is_a_singleton(self):
        approval.set_gate(None)
        assert approval.get_gate() is approval.get_gate()
        approval.set_gate(None)


class TestSerialisation:
    def test_a_request_renders_everything_the_ui_needs(self, gate):
        data = gate.submit(pause()).to_dict()
        for field in ("id", "status", "risk", "summary", "seconds_remaining", "original_query"):
            assert field in data
