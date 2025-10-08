"""
Answer policy: which provider runs, what happens when it does not, and what it cost.

The providers in ``agents/providers.py`` are deliberately naive — they call
their backend and raise if it fails. Deciding what a failure *means* is a
product question, not a provider question, and it lives here:

* a provider that is not configured is not an error, it is a different
  provider (``extractive``);
* a provider that fails mid-request falls back to extractive composition
  rather than returning an error string. An error string carries no citation,
  so the critic would mark the turn ungrounded and the operator would get a
  stack-trace fragment instead of the documents they asked about;
* a provider that streams and fails *after* emitting is not restarted —
  appending a second answer to a half-rendered one produces two answers
  stitched together. The turn is marked as cut short instead.

**One hosted call per question, and the caller gets the text back.** The UI
used to invoke the graph (one call) and then stream a fresh answer (a second
call), paying twice and displaying text the critic had never inspected.
``compose`` returns an ``AnswerOutcome`` carrying the text, the provider that
produced it, and the cost — so the UI has everything it needs from one call and
has no reason to make another.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agents import config, cost
from agents.providers import (
    AnswerProvider,
    AnswerRequest,
    get_answer_provider,
    resolve_answer_provider,
)

logger = logging.getLogger(__name__)

#: Shown in place of the rest of the answer when a stream dies part-way.
STREAM_CUT_SHORT = "\n\n_[The answer was cut short: the language model stream failed.]_"


@dataclass(frozen=True)
class AnswerOutcome:
    """One composed answer, with how it was produced."""

    text: str
    provider: str
    cost: cost.TurnCost
    fell_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "fell_back": self.fell_back,
            "cost": self.cost.to_dict(),
        }


def _request(
    query: str,
    contexts: list[dict],
    temperature: float,
    max_tokens: int,
    style: str | None,
) -> AnswerRequest:
    return AnswerRequest(
        query=query,
        contexts=contexts,
        style=style,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extractive() -> AnswerProvider:
    return get_answer_provider("extractive")


def _cost_for(provider: AnswerProvider, request: AnswerRequest, text: str) -> cost.TurnCost:
    """Prefer the provider's own token count; estimate when it gives none."""
    reported = cost.from_usage(getattr(provider, "last_usage", lambda: None)(), provider=provider.name)
    if reported is not None:
        return reported
    return cost.estimate_turn(
        request.query,
        request.contexts,
        answer=text,
        style=request.style,
        provider=provider.name,
    )


def compose(
    query: str,
    contexts: list[dict],
    temperature: float = 0.2,
    max_tokens: int = config.MAX_ANSWER_TOKENS,
    style: str | None = None,
) -> AnswerOutcome:
    """Compose one answer over the retrieved contexts. Exactly one hosted call."""
    request = _request(query, contexts, temperature, max_tokens, style)
    provider = resolve_answer_provider()

    if provider.name == "extractive":
        text = provider.complete(request)
        return AnswerOutcome(text=text, provider=provider.name, cost=_cost_for(provider, request, text))

    try:
        text = provider.complete(request)
    except Exception:
        logger.exception("Provider %r failed; falling back to extractive composition", provider.name)
        return _fallback(request, provider.name)

    if not text:
        logger.warning("Provider %r returned nothing; falling back to extractive composition", provider.name)
        return _fallback(request, provider.name)

    return AnswerOutcome(text=text, provider=provider.name, cost=_cost_for(provider, request, text))


def _fallback(request: AnswerRequest, failed_provider: str) -> AnswerOutcome:
    provider = _extractive()
    text = provider.complete(request)
    return AnswerOutcome(
        text=text,
        provider=f"{provider.name} (after {failed_provider} failed)",
        cost=_cost_for(provider, request, text),
        fell_back=True,
    )


def answer_with_citations(
    query: str,
    contexts: list[dict],
    temperature: float = 0.2,
    max_tokens: int = config.MAX_ANSWER_TOKENS,
    style: str | None = None,
) -> str:
    """The text of one composed answer. Convenience wrapper over ``compose``."""
    return compose(query, contexts, temperature=temperature, max_tokens=max_tokens, style=style).text


def stream_answer_with_citations(
    query: str,
    contexts: list[dict],
    temperature: float = 0.2,
    max_tokens: int = config.MAX_ANSWER_TOKENS,
    style: str | None = None,
) -> Iterator[str]:
    """Yield answer fragments in order.

    The caller accumulates them into the full response; the critic then runs
    over exactly that accumulated text, so what the operator read is what was
    checked. Used by the CLI and the eval harness; the Streamlit UI replays the
    already-composed answer rather than calling this, because streaming a
    second answer is what caused the double billing.
    """
    request = _request(query, contexts, temperature, max_tokens, style)
    provider = resolve_answer_provider()
    if provider.name == "extractive":
        yield from provider.stream(request)
        return

    emitted = False
    try:
        for fragment in provider.stream(request):
            emitted = True
            yield fragment
    except Exception:
        logger.exception("Provider %r stream failed", provider.name)
        if emitted:
            yield STREAM_CUT_SHORT
            return

    if not emitted:
        logger.warning("Provider %r streamed nothing; falling back to extractive composition", provider.name)
        yield from _extractive().stream(request)
