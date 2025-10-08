"""
HTML fragments for the operator console.

Kept out of ``streamlit_app.py`` so they can be imported and tested without
running a Streamlit script. Every fragment here ends up in
``st.markdown(..., unsafe_allow_html=True)``, and much of what it wraps is not
ours: retrieved passages come from staff-uploaded PDFs, the original query is
whatever the operator typed, and extractive answers quote the PDFs verbatim.
So every piece of text is escaped before it is placed inside markup. A PDF
containing ``<img src=x onerror=...>`` must render as text, not run.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from html import escape

from agents.grounding import CITATION_RE

REPLAY_CHUNK = 28
REPLAY_DELAY = 0.008


def replay(text: str, chunk_size: int = REPLAY_CHUNK, delay: float = REPLAY_DELAY) -> Iterator[str]:
    """Reveal already-composed, already-critiqued text progressively."""
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
        if delay:
            time.sleep(delay)


def pill(text: str, colour: str) -> str:
    return (
        f"<span style='background:{colour}1f;color:{colour};border:1px solid {colour}55;"
        f"border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:600;"
        f"white-space:nowrap;'>{escape(text)}</span>"
    )


def highlight_citations(text: str) -> str:
    """Escape the answer, then make ``[file pN]`` visually distinct from the prose around it.

    ``quote=False`` keeps apostrophes and quotes readable in the Markdown
    that Streamlit renders around the spans.
    """
    return CITATION_RE.sub(
        lambda m: f"<span class='cite'>{m.group(1)} p{m.group(2)}</span>",
        escape(text or "", quote=False),
    )


def source_card(position: int, source: str, badge: str, similarity_text: str, body: str) -> str:
    """One retrieved passage. ``badge`` is a ``pill`` and is already escaped."""
    return (
        f"<div class='src'><div class='src-head'>[{position}] {escape(source)} &nbsp;{badge} "
        f"<span class='muted'>&nbsp;{escape(similarity_text)}</span></div>"
        f"<div class='src-body'>{escape(body)}…</div></div>"
    )


def request_card(
    *,
    status_pill: str,
    risk_pill: str,
    request_id: str,
    summary: str,
    requested_by: str,
    original_query: str,
    destination: str,
    endpoint: str,
    colour: str,
) -> str:
    """The pending (or decided) action, as the approver reads it."""
    return (
        f"<div class='panel' style='border-color:{colour}66;'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;'>"
        f"{status_pill}{risk_pill}<span class='muted'>id {escape(request_id)}</span></div>"
        f"<div class='request-summary'>{escape(summary)}</div>"
        f"<div class='muted'>"
        f"Requested by <b>{escape(requested_by)}</b> from “{escape(original_query)}”.<br>"
        f"{escape(destination)} <code>{escape(endpoint)}</code>, "
        f"taken from the action registry — never from the query or the model."
        f"</div></div>"
    )
