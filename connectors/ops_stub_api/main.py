"""
The ops action API: the only component in this system that changes anything.

It is a stub of a real operations backend, and it is deliberately dumb. It
does not decide whether an action is allowed, does not parse intent, and has
never heard of a language model — it applies a named change and writes an
audit line. All the judgement lives upstream in ``agents/approval.py``.

That separation is the point. The agent can only reach this API through the
approval gate, the gate only sends endpoints that appear in the action
registry, and every request carries the ``approval_id`` of the human decision
that authorised it — so the audit log answers "who approved this, and when"
rather than only "what happened".
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# Configurable so the state and audit log can live on the same shared volume
# as the index; hardcoding a relative "data" meant state was written beside
# whatever directory the process happened to be started from.
DATA_DIR = Path(os.environ.get("OPS_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "ops_state.json"
AUDIT_FILE = DATA_DIR / "audit.log"

# Read-modify-write on a single file from a threaded server: two approvals
# arriving together would otherwise lose one of them.
_state_lock = threading.Lock()


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept flexible fields from UI

    type: str
    operator: str
    item: str | None = None
    details: str | None = None
    original_query: str | None = None
    #: The approval gate's request id. Optional so the endpoints stay usable
    #: with curl, but present on everything the agent sends — it is what joins
    #: this audit log to the decision that authorised the change.
    approval_id: str | None = None


class OpsState(BaseModel):
    model_config = ConfigDict(extra="allow")

    items: dict[str, str] = {}  # item -> status ("paused"|"active")
    hours_note: str = ""


def load_state() -> OpsState:
    if STATE_FILE.exists():
        try:
            return OpsState(**json.loads(STATE_FILE.read_text()))
        except Exception:
            pass
    return OpsState()


def save_state(state: OpsState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a crash midway through a write would otherwise leave
    # a truncated JSON file that load_state silently discards.
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    tmp.replace(STATE_FILE)


def append_audit(entry: dict[str, Any]) -> None:
    # `datetime.utcnow()` is deprecated from Python 3.12 and returns a naive
    # value that then gets a "Z" bolted on; this is the same instant, marked.
    entry = {**entry, "ts": datetime.now(UTC).isoformat()}
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


app = FastAPI(
    title="Ops Action API (stub)",
    version="1.0.0",
    description=__doc__,
)


@app.get("/health", tags=["ops"])
def health():
    return {"ok": True}


@app.get("/actions", tags=["ops"])
def list_actions() -> dict[str, list[dict[str, str]]]:
    """The action registry, as the agent sees it.

    Served from ``agents.actions`` rather than restated here, so this endpoint
    cannot drift from what the agent is actually able to propose.
    """
    from agents.actions import registry_rows

    return {"actions": registry_rows()}


@app.post("/pause_item", tags=["ops"])
def pause_item(payload: ActionPayload):
    item = (payload.item or "").strip()
    if not item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    with _state_lock:
        state = load_state()
        state.items[item] = "paused"
        save_state(state)
        append_audit({"action": "pause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Paused item '{item}'", "state": state.model_dump()}


@app.post("/unpause_item", tags=["ops"])
def unpause_item(payload: ActionPayload):
    item = (payload.item or "").strip()
    if not item:
        raise HTTPException(status_code=400, detail="Missing 'item' in payload")
    with _state_lock:
        state = load_state()
        state.items[item] = "active"
        save_state(state)
        append_audit({"action": "unpause_item", **payload.model_dump()})
    return {"ok": True, "message": f"Unpaused item '{item}'", "state": state.model_dump()}


@app.post("/update_hours", tags=["ops"])
def update_hours(payload: ActionPayload):
    note = payload.details or payload.original_query or "updated hours"
    with _state_lock:
        state = load_state()
        state.hours_note = note
        save_state(state)
        append_audit({"action": "update_hours", **payload.model_dump()})
    return {"ok": True, "message": "Hours updated", "note": note, "state": state.model_dump()}


@app.get("/state", tags=["ops"])
def get_state():
    return load_state()


@app.get("/audit", tags=["ops"])
def get_audit(limit: int = 100):
    """The audit trail, newest last.

    Paginated because it grows without bound: an unbounded read of a log file
    is fine on day one and a memory problem on day four hundred.
    """
    entries: list[dict[str, Any]] = []
    if AUDIT_FILE.exists():
        with AUDIT_FILE.open() as handle:
            for line in handle:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    # A partially written final line is expected if the process
                    # was killed mid-append; skip it rather than 500.
                    continue
    limit = max(1, min(limit, 1000))
    return {"count": len(entries), "returned": len(entries[-limit:]), "entries": entries[-limit:]}
