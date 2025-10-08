"""
Token and cost accounting for one question.

Worth a module because the headline finding of the last audit was a *cost* bug,
not a correctness one: the UI composed an answer through the graph and then
called the model a second time to stream a fresh one. Every question was billed
twice, and the answer the operator read was the second one — which the critic
had never seen.

Being able to say "a question costs this much" is also what makes the
chunk-size and ``top_k`` decisions arguable rather than arbitrary. Both of them
multiply straight into prompt tokens, and this is where that becomes a number.

Estimation, not measurement. The exact count comes back on the provider's
response and is used when present; this is the fallback and the planning tool.
Four characters per token is the usual rule of thumb for English prose and is
within a few percent on this corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agents import config
from agents.prompting import build_prompt, system_message

#: Characters per token. English prose, GPT-family tokenisers.
CHARS_PER_TOKEN = 4.0

#: USD per million tokens. Groq's published rate for the default model at the
#: time of writing; override with GROQ_INPUT_USD_PER_MTOK / _OUTPUT_ if it moves.
INPUT_USD_PER_MTOK = config.env_float("GROQ_INPUT_USD_PER_MTOK", 0.10)
OUTPUT_USD_PER_MTOK = config.env_float("GROQ_OUTPUT_USD_PER_MTOK", 0.50)


def estimate_tokens(text: str) -> int:
    """Rough token count for a string."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class TurnCost:
    """What one question cost, or would cost."""

    calls: int
    input_tokens: int
    output_tokens: int
    provider: str = "groq"

    @property
    def usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * INPUT_USD_PER_MTOK
            + self.output_tokens / 1_000_000 * OUTPUT_USD_PER_MTOK
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 8),
        }


def estimate_turn(
    query: str,
    contexts: Sequence[dict],
    answer: str = "",
    calls: int = 1,
    style: str | None = None,
    provider: str = "groq",
) -> TurnCost:
    """Estimate the cost of answering ``query`` over ``contexts``.

    ``calls`` is the multiplier that made the audit finding expensive: the same
    prompt sent twice costs twice, and passing ``calls=2`` reproduces the old
    behaviour for comparison.
    """
    if provider == "extractive":
        # No hosted call at all. Saying "$0" explicitly is the point of the
        # keyless mode, so it is a first-class result rather than a missing one.
        return TurnCost(calls=0, input_tokens=0, output_tokens=0, provider=provider)
    prompt_tokens = estimate_tokens(system_message(style)) + estimate_tokens(build_prompt(query, list(contexts)))
    completion_tokens = estimate_tokens(answer) if answer else config.MAX_ANSWER_TOKENS // 2
    return TurnCost(
        calls=calls,
        input_tokens=prompt_tokens * calls,
        output_tokens=completion_tokens * calls,
        provider=provider,
    )


def from_usage(usage: Any, calls: int = 1, provider: str = "groq") -> TurnCost | None:
    """Build a ``TurnCost`` from a provider's own usage object, if it gave one."""
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
    if prompt is None:
        return None
    return TurnCost(
        calls=calls,
        input_tokens=int(prompt),
        output_tokens=int(completion or 0),
        provider=provider,
    )


def total(costs: Sequence[TurnCost]) -> TurnCost:
    """Sum a run's worth of turns."""
    costs = list(costs)
    if not costs:
        return TurnCost(calls=0, input_tokens=0, output_tokens=0)
    return TurnCost(
        calls=sum(c.calls for c in costs),
        input_tokens=sum(c.input_tokens for c in costs),
        output_tokens=sum(c.output_tokens for c in costs),
        provider=costs[0].provider,
    )


def breakdown(query: str, contexts: Sequence[dict], style: str | None = None) -> list[dict[str, Any]]:
    """Where the prompt tokens go. Rendered in the UI's cost panel.

    Retrieved context dominates — which is the argument for capping it, and the
    reason ``top_k`` is a cost control rather than a quality dial.
    """
    system = system_message(style)
    context_prompt = build_prompt(query, list(contexts))
    bare_prompt = build_prompt(query, [])
    context_only = max(0, estimate_tokens(context_prompt) - estimate_tokens(bare_prompt))
    return [
        {"part": "system prompt", "tokens": estimate_tokens(system)},
        {"part": "retrieved context", "tokens": context_only},
        {"part": "question", "tokens": estimate_tokens(bare_prompt)},
    ]
