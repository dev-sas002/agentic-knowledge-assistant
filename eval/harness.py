"""
An offline eval harness over a fixed question set.

A RAG system has no unit test for "is the answer good", and the alternative to
measuring is arguing. This runs every question in ``eval/questions.json``
through the real graph and reports four things per question:

* **retrieval hit** — did the expected document appear in the retrieved set?
  This separates a retrieval failure from a composition failure, which is the
  single most useful distinction when a RAG answer is wrong.
* **grounded** — the critic's verdict, from ``agents/grounding.py``.
* **latency** — embed and query milliseconds, so the chunking and ``top_k``
  decisions can be argued with numbers.
* **cost** — estimated tokens and USD per question.

It is deliberately runnable with no API key: the extractive provider answers
from retrieved text, so the harness measures retrieval and grounding without
spending anything. With ``GROQ_API_KEY`` set it measures the hosted path
instead, and the cost column becomes real.

The set includes one question the corpus cannot answer. A harness that only
asks answerable questions rewards a system that answers everything
confidently, which is the failure mode that matters most here.

    python -m eval.harness              # table
    python -m eval.harness --json       # machine-readable
    python -m eval.harness --top-k 6    # sweep the fan-out
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import cost
from agents.graph import invoke_graph

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


@dataclass
class QuestionResult:
    """One question's outcome."""

    id: str
    query: str
    retrieval_hit: bool
    grounded: bool
    grounding_score: float
    sources: list[str]
    latency_ms: float
    retrieval_ms: float
    phrases_found: int
    phrases_expected: int
    turn_cost: cost.TurnCost
    provider: str
    expected_ungrounded: bool = False

    @property
    def passed(self) -> bool:
        """What "correct" means for this question.

        For an answerable question: the expected document was retrieved and the
        critic accepted the answer. For the unanswerable one, the opposite —
        declining to answer is the pass condition.
        """
        if self.expected_ungrounded:
            return not self.grounded
        return self.retrieval_hit and self.grounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "passed": self.passed,
            "retrieval_hit": self.retrieval_hit,
            "grounded": self.grounded,
            "grounding_score": round(self.grounding_score, 3),
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "phrases": f"{self.phrases_found}/{self.phrases_expected}",
            "provider": self.provider,
            "cost": self.turn_cost.to_dict(),
        }


@dataclass
class EvalReport:
    """The whole run."""

    results: list[QuestionResult] = field(default_factory=list)
    top_k: int = 4
    style: str = "detailed"

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def retrieval_recall(self) -> float:
        answerable = [r for r in self.results if not r.expected_ungrounded]
        if not answerable:
            return 0.0
        return sum(1 for r in answerable if r.retrieval_hit) / len(answerable)

    @property
    def grounded_rate(self) -> float:
        answerable = [r for r in self.results if not r.expected_ungrounded]
        if not answerable:
            return 0.0
        return sum(1 for r in answerable if r.grounded) / len(answerable)

    @property
    def median_latency_ms(self) -> float:
        return statistics.median([r.latency_ms for r in self.results]) if self.results else 0.0

    @property
    def median_retrieval_ms(self) -> float:
        return statistics.median([r.retrieval_ms for r in self.results]) if self.results else 0.0

    @property
    def total_cost(self) -> cost.TurnCost:
        return cost.total([r.turn_cost for r in self.results])

    def to_dict(self) -> dict[str, Any]:
        per_question = self.total_cost.usd / len(self.results) if self.results else 0.0
        return {
            "top_k": self.top_k,
            "style": self.style,
            "questions": len(self.results),
            "passed": self.passed,
            "retrieval_recall": round(self.retrieval_recall, 3),
            "grounded_rate": round(self.grounded_rate, 3),
            "median_latency_ms": round(self.median_latency_ms, 1),
            "median_retrieval_ms": round(self.median_retrieval_ms, 1),
            "cost_total": self.total_cost.to_dict(),
            "cost_per_question_usd": round(per_question, 8),
            "results": [r.to_dict() for r in self.results],
        }


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["questions"]


def _sources_of(contexts: list[dict]) -> list[str]:
    seen: list[str] = []
    for context in contexts:
        name = (context.get("metadata") or {}).get("filename", "unknown")
        if name not in seen:
            seen.append(name)
    return seen


def run_question(question: dict[str, Any], top_k: int = 4, style: str = "detailed") -> QuestionResult:
    started = time.perf_counter()
    state = invoke_graph(query=question["query"], top_k=top_k, style=style)
    elapsed = (time.perf_counter() - started) * 1000

    contexts = state.get("contexts", []) or []
    sources = _sources_of(contexts)
    expected = question.get("expect_sources") or []
    retrieval_hit = (not expected) or any(name in sources for name in expected)

    answer = state.get("answer", "") or ""
    phrases = question.get("expect_phrases") or []
    found = sum(1 for phrase in phrases if phrase.lower() in answer.lower())

    report = state.get("grounding", {}) or {}
    meta = state.get("answer_meta", {}) or {}
    cost_dict = meta.get("cost") or {}
    turn_cost = cost.TurnCost(
        calls=cost_dict.get("calls", 0),
        input_tokens=cost_dict.get("input_tokens", 0),
        output_tokens=cost_dict.get("output_tokens", 0),
        provider=cost_dict.get("provider", "unknown"),
    )
    retrieval = state.get("retrieval", {}) or {}

    return QuestionResult(
        id=question["id"],
        query=question["query"],
        retrieval_hit=retrieval_hit,
        grounded=bool(state.get("grounded")),
        grounding_score=float(report.get("score", 0.0)),
        sources=sources,
        latency_ms=elapsed,
        retrieval_ms=float(retrieval.get("embed_ms", 0.0)) + float(retrieval.get("query_ms", 0.0)),
        phrases_found=found,
        phrases_expected=len(phrases),
        turn_cost=turn_cost,
        provider=meta.get("provider", "unknown"),
        expected_ungrounded=bool(question.get("expect_ungrounded")),
    )


def run(questions: list[dict[str, Any]] | None = None, top_k: int = 4, style: str = "detailed") -> EvalReport:
    questions = questions if questions is not None else load_questions()
    report = EvalReport(top_k=top_k, style=style)
    for question in questions:
        report.results.append(run_question(question, top_k=top_k, style=style))
    return report


def format_table(report: EvalReport) -> str:
    header = f"{'id':<20} {'pass':<5} {'retr':<5} {'grnd':<5} {'score':<6} {'ms':>7}  sources"
    lines = [header, "-" * len(header)]
    for result in report.results:
        lines.append(
            f"{result.id:<20} "
            f"{'✓' if result.passed else '✗':<5} "
            f"{'✓' if result.retrieval_hit else '·':<5} "
            f"{'✓' if result.grounded else '·':<5} "
            f"{result.grounding_score:<6.2f} "
            f"{result.latency_ms:>7.0f}  "
            f"{', '.join(result.sources[:2]) or '—'}"
        )
    summary = report.to_dict()
    lines += [
        "-" * len(header),
        f"{report.passed}/{len(report.results)} passed   "
        f"retrieval recall {report.retrieval_recall:.0%}   "
        f"grounded {report.grounded_rate:.0%}",
        f"median latency {report.median_latency_ms:.0f}ms "
        f"(retrieval {report.median_retrieval_ms:.0f}ms)   "
        f"provider {report.results[0].provider if report.results else 'n/a'}",
        f"cost {summary['cost_total']['calls']} call(s), "
        f"{summary['cost_total']['total_tokens']} tokens, "
        f"${summary['cost_total']['usd']:.6f} total — "
        f"${summary['cost_per_question_usd']:.6f}/question",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline eval harness.")
    parser.add_argument("--top-k", type=int, default=4, help="Passages retrieved per question")
    parser.add_argument("--style", default="detailed", choices=["detailed", "concise"])
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument("--fail-under", type=float, default=0.0, help="Exit non-zero below this pass rate")
    args = parser.parse_args(argv)

    report = run(top_k=args.top_k, style=args.style)
    print(json.dumps(report.to_dict(), indent=2) if args.json else format_table(report))

    rate = report.passed / len(report.results) if report.results else 0.0
    if rate < args.fail_under:
        print(f"\nPass rate {rate:.0%} is below --fail-under {args.fail_under:.0%}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
