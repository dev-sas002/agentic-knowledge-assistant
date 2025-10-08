"""
The critic: does the answer actually rest on what was retrieved?

The original check was ``bool(re.search(r"\\[[^\\]]+ p\\d+\\]", answer))`` — one
citation anywhere means grounded. That passes an answer of nine invented
sentences and one citation, and it passes a citation to ``budget_2019.pdf p3``
when no such document was retrieved. Both are the failures a RAG system exists
to prevent, so the check is worth more than one regex.

Three signals, each cheap and each checking something different:

* **Coverage** — what fraction of the answer's substantive sentences carry a
  citation. Low coverage means most of the answer is unsourced prose.
* **Validity** — what fraction of the citations name a source that was
  actually retrieved this turn. This is the one that catches a model inventing
  a plausible-looking filename, and it is why the retrieved set is passed in
  rather than the answer being checked on its own.
* **Support** — whether the cited passage shares vocabulary with the sentence
  citing it. Lexical, so it is a weak signal and is weighted as one; it exists
  to catch a citation attached to the wrong retrieved passage.

What this is not: an entailment model. It cannot tell you that a correctly
cited sentence misstates its source. It is a cheap, deterministic, explainable
floor that runs on every turn with no key and no second model call — and the
report it returns names which sentences failed, so the operator can judge.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

#: ``[filename pN]`` — the citation shape the whole system agrees on.
CITATION_RE = re.compile(r"\[([^\]]+?) p(\d+|\?)\]")

#: Sentences shorter than this are headings, list labels or transitions, and
#: demanding a citation on them punishes well-structured answers.
_MIN_SUBSTANTIVE_CHARS = 40

#: The fraction of substantive sentences that must carry a citation before an
#: answer counts as grounded, regardless of how good the other signals are.
MIN_COVERAGE = 0.4

#: Sentence splitting, with the citation kept attached to the claim it
#: supports. Three alternatives, in order: a sentence end that is *not*
#: followed by a citation; the end of a trailing citation; a line break.
#:
#: The naive `(?<=[.!?])\s+` split "…for the preceding week. [finance.pdf p7]"
#: into an uncited claim plus a bare citation, which scored every properly
#: cited answer at 50% coverage.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?!\[)|(?<=\])[ \t]+(?=[A-Z0-9])|\n+")

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "then", "there", "these", "this", "to", "was", "were", "which", "with",
}


@dataclass(frozen=True)
class GroundingReport:
    """The critic's verdict, with its working shown."""

    grounded: bool
    score: float
    coverage: float
    validity: float
    support: float
    citations: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    reason: str = ""

    def to_state(self) -> dict:
        """Flat dict for graph state and for the UI."""
        return {
            "grounded": self.grounded,
            "score": round(self.score, 3),
            "coverage": round(self.coverage, 3),
            "validity": round(self.validity, 3),
            "support": round(self.support, 3),
            "citations": self.citations,
            "unknown_citations": self.unknown_citations,
            "cited_sources": self.cited_sources,
            "uncited_sentences": self.uncited_sentences,
            "reason": self.reason,
        }


def source_key(filename: str, page) -> str:
    return f"{str(filename).strip().lower()} p{page}"


def context_sources(contexts: Sequence[dict]) -> set[str]:
    """Every ``filename pN`` that was actually retrieved this turn."""
    sources = set()
    for context in contexts or ():
        meta = context.get("metadata") or {}
        filename = meta.get("filename") or meta.get("source") or "unknown"
        sources.add(source_key(filename, meta.get("page", "?")))
    return sources


def citations(text: str) -> list[str]:
    """Every citation in the text, in order, deduplicated."""
    found = [source_key(match.group(1), match.group(2)) for match in CITATION_RE.finditer(text or "")]
    return list(dict.fromkeys(found))


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]
    return [part for part in parts if part]


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall((text or "").lower()) if word not in _STOPWORDS and len(word) > 2}


def _is_substantive(sentence: str) -> bool:
    """Is this a claim that ought to carry a citation?

    Scaffolding is not. Demanding a citation on a heading, a lead-in ending in
    a colon, a table row, or the composer's own italicised footer about how the
    answer was produced would punish a well-structured answer for being
    well-structured.
    """
    text = sentence.strip()
    if text.endswith(":"):  # a lead-in, not a claim
        return False
    if len(text) > 1 and text[0] in "*_" and text[-1] in "*_":  # wholly emphasised
        return False
    stripped = text.strip("*_# ")
    return len(stripped) >= _MIN_SUBSTANTIVE_CHARS and not stripped.startswith(("|", ">"))


def _support_for(sentence: str, cited: list[str], by_source: dict[str, str]) -> float:
    """Vocabulary overlap between one sentence and the passages it cites."""
    sentence_tokens = _tokens(CITATION_RE.sub("", sentence))
    if not sentence_tokens:
        return 1.0
    best = 0.0
    for key in cited:
        passage_tokens = _tokens(by_source.get(key, ""))
        if not passage_tokens:
            continue
        best = max(best, len(sentence_tokens & passage_tokens) / len(sentence_tokens))
    return best


def assess(answer: str, contexts: Sequence[dict], threshold: float = 0.45) -> GroundingReport:
    """Score how well ``answer`` is grounded in ``contexts``."""
    answer = answer or ""
    known = context_sources(contexts)
    by_source: dict[str, str] = {}
    for context in contexts or ():
        meta = context.get("metadata") or {}
        key = source_key(meta.get("filename") or meta.get("source") or "unknown", meta.get("page", "?"))
        # Several chunks can share a page; join them so a citation is checked
        # against everything retrieved from that page.
        by_source[key] = (by_source.get(key, "") + " " + (context.get("document") or "")).strip()

    found = citations(answer)
    if not found:
        return GroundingReport(
            grounded=False,
            score=0.0,
            coverage=0.0,
            validity=0.0,
            support=0.0,
            reason="The answer cites no sources.",
        )

    unknown = [key for key in found if key not in known]
    validity = (len(found) - len(unknown)) / len(found)

    substantive = [s for s in sentences(answer) if _is_substantive(s)]
    cited_sentences = [s for s in substantive if CITATION_RE.search(s)]
    uncited = [s for s in substantive if not CITATION_RE.search(s)]
    coverage = len(cited_sentences) / len(substantive) if substantive else 0.0

    supports = [
        _support_for(sentence, citations(sentence), by_source)
        for sentence in cited_sentences
    ]
    support = sum(supports) / len(supports) if supports else 0.0

    # The score is a weighted blend for reporting, but "grounded" is not a
    # threshold on it. Two conditions gate the verdict outright, because a
    # weighted average lets a strong signal paper over a fatal one:
    #
    #   * any fabricated citation fails the turn, however well-written the
    #     rest of it is;
    #   * nine invented sentences plus one correct citation is not a grounded
    #     answer, even though validity is a perfect 1.0.
    score = (0.5 * validity) + (0.3 * coverage) + (0.2 * support)
    grounded = not unknown and coverage >= MIN_COVERAGE and score >= threshold

    if unknown:
        reason = f"{len(unknown)} citation(s) name sources that were not retrieved: {', '.join(unknown)}."
    elif coverage < MIN_COVERAGE:
        reason = f"Only {len(cited_sentences)} of {len(substantive)} substantive sentences carry a citation."
    elif grounded:
        reason = f"{len(found)} citation(s), all resolving to retrieved sources."
    else:
        reason = "Cited passages share little vocabulary with the sentences citing them."

    return GroundingReport(
        grounded=grounded,
        score=score,
        coverage=coverage,
        validity=validity,
        support=support,
        citations=found,
        unknown_citations=unknown,
        cited_sources=[key for key in found if key in known],
        uncited_sentences=uncited,
        reason=reason,
    )
