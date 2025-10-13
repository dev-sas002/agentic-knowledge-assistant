"""
Tests for the ops action stub.

This is the component that "executes" an approved action and writes the audit
record, so the things worth pinning down are: an action changes state, an
approval is always audited, a malformed payload is refused rather than
recorded, and a corrupted state file does not take the service down.

Every test runs against a temporary directory; nothing touches `data/`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import connectors.ops_stub_api.main as ops


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ops, "STATE_FILE", tmp_path / "ops_state.json")
    monkeypatch.setattr(ops, "AUDIT_FILE", tmp_path / "audit.log")
    return TestClient(ops.app)


def audit_entries(client) -> list[dict]:
    return client.get("/audit").json()["entries"]


class TestHealth:
    def test_health_reports_ok(self, client):
        assert client.get("/health").json() == {"ok": True}


class TestPausing:
    def test_pausing_an_item_records_it_as_paused(self, client):
        r = client.post(
            "/pause_item", json={"type": "pause_item", "operator": "alice", "item": "Chicken Biryani"}
        )
        assert r.status_code == 200
        assert r.json()["state"]["items"]["Chicken Biryani"] == "paused"
        assert client.get("/state").json()["items"]["Chicken Biryani"] == "paused"

    def test_unpausing_reverses_it(self, client):
        payload = {"type": "pause_item", "operator": "alice", "item": "Soup"}
        client.post("/pause_item", json=payload)
        client.post("/unpause_item", json={**payload, "type": "unpause_item"})
        assert client.get("/state").json()["items"]["Soup"] == "active"

    @pytest.mark.parametrize("item", [None, "", "   "])
    def test_an_action_with_no_item_is_refused(self, client, item):
        # Better a 400 than an audit entry that says an empty thing was paused.
        body = {"type": "pause_item", "operator": "alice"}
        if item is not None:
            body["item"] = item
        assert client.post("/pause_item", json=body).status_code == 400

    def test_a_refused_action_is_not_audited(self, client):
        client.post("/pause_item", json={"type": "pause_item", "operator": "alice", "item": ""})
        assert audit_entries(client) == []

    def test_a_missing_operator_is_a_validation_error(self, client):
        # The whole safety story is "a human approved it"; an unattributed
        # action must not be accepted.
        assert client.post("/pause_item", json={"type": "pause_item", "item": "Soup"}).status_code == 422


class TestHours:
    def test_details_become_the_note(self, client):
        r = client.post(
            "/update_hours",
            json={"type": "update_hours", "operator": "bob", "details": "set hours to 10:00-22:00"},
        )
        assert r.json()["note"] == "set hours to 10:00-22:00"

    def test_the_original_query_is_used_when_no_details_were_parsed(self, client):
        r = client.post(
            "/update_hours",
            json={"type": "update_hours", "operator": "bob", "original_query": "open later on Fridays"},
        )
        assert r.json()["note"] == "open later on Fridays"

    def test_there_is_always_a_note(self, client):
        r = client.post("/update_hours", json={"type": "update_hours", "operator": "bob"})
        assert r.json()["note"]


class TestAudit:
    def test_every_executed_action_is_audited(self, client):
        client.post("/pause_item", json={"type": "pause_item", "operator": "alice", "item": "Soup"})
        client.post("/update_hours", json={"type": "update_hours", "operator": "bob", "details": "x"})
        actions = [e["action"] for e in audit_entries(client)]
        assert actions == ["pause_item", "update_hours"]

    def test_the_audit_records_who_approved_it(self, client):
        client.post("/pause_item", json={"type": "pause_item", "operator": "alice", "item": "Soup"})
        assert audit_entries(client)[0]["operator"] == "alice"

    def test_the_audit_records_when(self, client):
        client.post("/pause_item", json={"type": "pause_item", "operator": "alice", "item": "Soup"})
        entry = audit_entries(client)[0]
        assert entry["ts"]
        # Timezone-aware, unlike the naive utcnow() + "Z" it replaced.
        assert entry["ts"].endswith("+00:00")

    def test_the_audit_keeps_the_text_the_operator_actually_typed(self, client):
        client.post(
            "/pause_item",
            json={
                "type": "pause_item",
                "operator": "alice",
                "item": "Chicken Biryani",
                "original_query": "pause 'Chicken Biryani'",
            },
        )
        assert audit_entries(client)[0]["original_query"] == "pause 'Chicken Biryani'"

    def test_a_corrupt_audit_line_does_not_break_the_endpoint(self, client, tmp_path):
        client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": "Soup"})
        (tmp_path / "audit.log").open("a").write("not json at all\n")
        assert client.get("/audit").status_code == 200


class TestStatePersistence:
    def test_state_survives_a_fresh_client(self, client, tmp_path, monkeypatch):
        client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": "Soup"})
        second = TestClient(ops.app)
        assert second.get("/state").json()["items"]["Soup"] == "paused"

    def test_a_corrupt_state_file_falls_back_to_an_empty_state(self, client, tmp_path):
        (tmp_path / "ops_state.json").write_text("{ this is not json")
        assert client.get("/state").json()["items"] == {}

    def test_the_state_file_is_valid_json_after_a_write(self, client, tmp_path):
        client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": "Soup"})
        assert json.loads((tmp_path / "ops_state.json").read_text())["items"] == {"Soup": "paused"}

    def test_no_temporary_file_is_left_behind(self, client, tmp_path):
        client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": "Soup"})
        assert not list(tmp_path.glob("*.tmp"))


class TestActionRegistryEndpoint:
    """The ops API publishes the same registry the agent proposes from.

    Restating the list here would let the two drift, and a drifted allow-list
    is the kind of gap that only shows up when something is approved and then
    404s.
    """

    def test_the_registry_is_served(self, client):
        actions = client.get("/actions").json()["actions"]
        assert {row["action"] for row in actions} == {"pause_item", "unpause_item", "update_hours"}

    def test_every_advertised_endpoint_actually_exists(self, client):
        paths = {route.path for route in ops.app.routes}
        for row in client.get("/actions").json()["actions"]:
            assert row["endpoint"] in paths

    def test_every_action_declares_a_risk_and_an_approval_requirement(self, client):
        for row in client.get("/actions").json()["actions"]:
            assert row["risk"] in {"low", "medium", "high"}
            assert row["approval"] == "required"


class TestApprovalProvenance:
    def test_the_approval_id_is_audited(self, client):
        # This is what joins the ops audit log to the human decision that
        # authorised the change.
        client.post(
            "/pause_item",
            json={"type": "pause_item", "operator": "alice", "item": "Soup", "approval_id": "abc123"},
        )
        assert audit_entries(client)[-1]["approval_id"] == "abc123"

    def test_an_action_without_an_approval_id_is_still_accepted(self, client):
        # curl against the stub must keep working; the field is optional.
        response = client.post("/pause_item", json={"type": "pause_item", "operator": "alice", "item": "Soup"})
        assert response.status_code == 200


class TestAuditPagination:
    def test_the_audit_is_bounded(self, client):
        # An unbounded read of a growing log file is fine on day one and a
        # memory problem on day four hundred.
        for index in range(30):
            client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": f"item-{index}"})
        body = client.get("/audit?limit=5").json()
        assert body["count"] == 30
        assert body["returned"] == 5

    def test_the_newest_entries_are_returned(self, client):
        for index in range(5):
            client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": f"item-{index}"})
        assert client.get("/audit?limit=2").json()["entries"][-1]["item"] == "item-4"

    def test_an_absurd_limit_is_clamped(self, client):
        client.post("/pause_item", json={"type": "pause_item", "operator": "a", "item": "x"})
        assert client.get("/audit?limit=99999").json()["returned"] == 1
