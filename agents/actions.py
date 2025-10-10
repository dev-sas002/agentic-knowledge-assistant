"""
The action registry: a closed allow-list of operational changes the agent may
*propose*.

Two properties matter more than the parsing itself.

**It is closed.** An action exists only if it is registered here, with a fixed
ops endpoint. The endpoint is never derived from model output or from parsed
user text, so no input can cause a request to somewhere unintended. Adding an
action is one ``ActionSpec``; there is no dynamic dispatch path to audit.

**It is driven by the operator's own words.** ``propose`` is given the query
the operator typed — never retrieved document text, never model output. A PDF
in the corpus therefore cannot reach this module at all, which is the
structural half of the prompt-injection defence described in
``agents/prompting.py``. The other half is that nothing here executes: every
spec returns a proposal, and ``agents/approval.py`` holds it for a human.

Naming the wrong item is the failure mode this module is built around. An
approval screen that reads correctly while acting on something else is worse
than no approval screen, so the parsers are conservative: they propose
``unknown`` rather than guess.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class Risk(str, Enum):
    """How much a mistake costs. Rendered on the approval screen."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ProposedAction:
    """A structured change, parsed but not executed."""

    type: str
    params: dict[str, str] = field(default_factory=dict)
    original_query: str = ""

    @property
    def is_known(self) -> bool:
        return self.type in ACTIONS

    def to_state(self) -> dict:
        """Flat dict for graph state and for the ops API payload."""
        return {"type": self.type, "original_query": self.original_query, **self.params}

    def describe(self) -> str:
        spec = ACTIONS.get(self.type)
        if spec is None:
            return "No recognised action."
        return spec.describe(self.params)

    @property
    def risk(self) -> Risk:
        spec = ACTIONS.get(self.type)
        return spec.risk if spec else Risk.LOW


#: What ``propose`` returns when nothing in the registry matched. Better to
#: propose nothing than to guess at an operational change.
def unknown_action(query: str) -> ProposedAction:
    return ProposedAction(type="unknown", original_query=query)


@dataclass(frozen=True)
class ActionSpec:
    """One registered action.

    ``endpoint`` is the single source of truth for where an approved action is
    sent. ``agents/approval.py`` reads it from here and nowhere else.
    """

    name: str
    description: str
    endpoint: str
    risk: Risk
    #: Substrings that make the router treat a query as a command rather than
    #: a question. Keeping them on the spec means registering an action also
    #: teaches the router about it.
    triggers: tuple[str, ...]
    #: Query text -> parameters, or None when this spec does not apply.
    parser: Callable[[str], dict[str, str] | None]
    #: Human-readable one-liner for the approval screen.
    summary: Callable[[dict[str, str]], str]
    requires_approval: bool = True

    def describe(self, params: dict[str, str]) -> str:
        return self.summary(params)


ACTIONS: dict[str, ActionSpec] = {}


def register_action(spec: ActionSpec) -> ActionSpec:
    ACTIONS[spec.name] = spec
    return spec


def action_triggers() -> tuple[str, ...]:
    """Every trigger word across the registry, for the router."""
    return tuple(trigger for spec in ACTIONS.values() for trigger in spec.triggers)


def looks_like_action(query: str) -> bool:
    lowered = (query or "").lower()
    return any(trigger in lowered for trigger in action_triggers())


def propose(query: str) -> ProposedAction:
    """Parse the operator's query into a proposed action, or ``unknown``."""
    query = query or ""
    for spec in ACTIONS.values():
        params = spec.parser(query)
        if params:
            return ProposedAction(type=spec.name, params=params, original_query=query)
    return unknown_action(query)


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
# Quoted name first, then bare text to the end of the line.
#
# The pattern this replaced was `(pause|unpause|resume)\s+'?"?([^'"]+?)'?"?\b`,
# whose non-greedy group stopped at the first word boundary: "pause 'Chicken
# Biryani'" extracted the item as "Chicken".
def _item_pattern(verbs: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?:{verbs})\s+(?:'([^']+)'|\"([^\"]+)\"|(.+?))\s*[.!?]*\s*$",
        re.IGNORECASE,
    )


_PAUSE_RE = _item_pattern("pause")
_UNPAUSE_RE = _item_pattern("unpause|resume")
_HOURS_RE = re.compile(r"(update|set|change)\s+(opening\s+)?hours(.*)", re.IGNORECASE)


def _extract_item(pattern: re.Pattern[str], query: str) -> dict[str, str] | None:
    match = pattern.search(query)
    if not match:
        return None
    # Exactly one of the quoted / double-quoted / bare groups is set.
    # `next(..., None)` rather than a bare `next()`: an unexpected regex
    # change must not raise StopIteration out of a graph node.
    raw = next((group for group in match.groups() if group), None)
    if raw is None:
        return None
    item = raw.strip().strip("'\"").strip()
    return {"item": item} if item else None


register_action(
    ActionSpec(
        name="pause_item",
        description="Take a menu item off sale.",
        endpoint="/pause_item",
        risk=Risk.MEDIUM,
        triggers=("pause",),
        parser=lambda q: _extract_item(_PAUSE_RE, q),
        summary=lambda p: f"Pause the menu item {p.get('item', '?')!r} (it stops being orderable).",
    )
)

register_action(
    ActionSpec(
        name="unpause_item",
        description="Put a paused menu item back on sale.",
        endpoint="/unpause_item",
        risk=Risk.LOW,
        triggers=("unpause", "resume"),
        parser=lambda q: _extract_item(_UNPAUSE_RE, q),
        summary=lambda p: f"Make the menu item {p.get('item', '?')!r} orderable again.",
    )
)


def _parse_hours(query: str) -> dict[str, str] | None:
    match = _HOURS_RE.search(query)
    if not match:
        return None
    return {"details": match.group(0).strip()}


register_action(
    ActionSpec(
        name="update_hours",
        description="Change published opening hours.",
        endpoint="/update_hours",
        risk=Risk.HIGH,
        triggers=("update hours", "change hours", "set hours", "update opening hours"),
        parser=_parse_hours,
        summary=lambda p: f"Change published opening hours: {p.get('details', '?')}.",
    )
)


def registry_rows() -> list[dict[str, str]]:
    """The registry as plain rows, for the UI's 'what can it do' panel."""
    return [
        {
            "action": spec.name,
            "description": spec.description,
            "risk": spec.risk.value,
            "approval": "required" if spec.requires_approval else "not required",
            "endpoint": spec.endpoint,
        }
        for spec in ACTIONS.values()
    ]
