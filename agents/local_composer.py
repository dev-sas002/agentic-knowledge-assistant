"""
Extractive answer composition, used when no GROQ_API_KEY is configured.

Without a key, `answer_with_citations` returned the literal string
"[LLM disabled] Provide GROQ_API_KEY in .env." — so the compose step produced
no answer, the critic's grounding check always failed, and the whole pipeline
was undemonstrable to anyone who cloned the repository. Routing, retrieval,
chunking, citation handling and the critic are the parts worth reviewing, and
none of them need a hosted model.

**Extractive on purpose.** A fake generative model would have to invent prose,
and inventing prose is precisely the failure a retrieval system exists to
prevent. Every sentence emitted here is present verbatim in a retrieved chunk,
so the citations point at text a reader can actually check — which means the
critic node is doing real work rather than rubber-stamping a canned string.

What this proves: routing, retrieval, ranking, citation formatting and the
grounding check all work end to end. What it does not prove: the quality of a
synthesised narrative across several sources. That needs a real model, which is
why this sits behind the same function signature rather than replacing it.
"""

from __future__ import annotations

import re

# Words too common to indicate that a sentence is about the query.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or",
    "our", "the", "to", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "should", "would", "could", "there", "this",
    "that", "these", "those", "have", "has", "had", "not", "no", "yes",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


#: Suffixes stripped before comparing words, longest first. This is not a
#: linguistic stemmer — it exists so that a question asking about a "payout"
#: matches a document that says "payouts", which is the single most common
#: reason an obviously relevant sentence scores zero.
_SUFFIXES = ("ing", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            base = word[: -len(suffix)]
            # "policies" -> "polic" -> "poli" reads badly; restore the y.
            return base + "y" if suffix == "ies" else base
    return word


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def _citation(meta: dict | None) -> str:
    """Format a source reference the critic recognises: [filename pN]."""
    meta = meta or {}
    filename = meta.get("filename") or meta.get("source") or "unknown"
    page = meta.get("page", "?")
    return f"[{filename} p{page}]"


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]
    # Very short fragments are usually headings or list bullets split badly.
    return [s for s in parts if len(s) > 25]


def score_sentence(sentence: str, query_tokens: set[str]) -> float:
    """
    Overlap between a sentence and the question.

    Normalised by sentence length so a long paragraph does not outrank a
    precise one simply by containing more words.
    """
    tokens = _tokens(sentence)
    if not tokens:
        return 0.0
    overlap = len(tokens & query_tokens)
    if overlap == 0:
        return 0.0
    return overlap / (len(tokens) ** 0.5)


def compose_extractive(
    query: str,
    contexts: list[dict],
    max_sentences: int = 6,
    style: str | None = None,
) -> str:
    """Build an answer from sentences that are actually in the sources."""
    if not contexts:
        return (
            "No supporting documents were retrieved for this question, so there "
            "is nothing to quote. Try rephrasing, or check that the index has "
            "been built (`python -m ingestion.build_index`)."
        )

    query_tokens = _tokens(query)

    scored: list[tuple[float, str, str]] = []
    for context in contexts:
        citation = _citation(context.get("metadata"))
        for sentence in _sentences(context.get("document") or ""):
            score = score_sentence(sentence, query_tokens)
            if score > 0:
                scored.append((score, sentence, citation))

    scored.sort(key=lambda row: row[0], reverse=True)

    if style and style.lower() == "concise":
        max_sentences = min(max_sentences, 3)

    # Two passes. The first takes the best sentence from each distinct source,
    # so the answer spans what was retrieved rather than quoting one chunk. The
    # second fills the remaining slots with the next best sentences, wherever
    # they came from.
    #
    # The second pass is not optional: retrieval frequently returns several
    # chunks from the same page, which share a citation. A single pass that
    # skipped repeated citations collapsed the whole answer to one sentence.
    chosen: list[tuple[str, str]] = []
    seen_sentences: set[str] = set()
    used_citations: set[str] = set()

    for _, sentence, citation in scored:
        if len(chosen) >= max_sentences:
            break
        if sentence in seen_sentences or citation in used_citations:
            continue
        chosen.append((sentence, citation))
        seen_sentences.add(sentence)
        used_citations.add(citation)

    for _, sentence, citation in scored:
        if len(chosen) >= max_sentences:
            break
        if sentence in seen_sentences:
            continue
        chosen.append((sentence, citation))
        seen_sentences.add(sentence)

    # Retrieval is semantic; this sentence selection is lexical. That gap is
    # real: "Payments are issued weekly on Fridays" answers "how are payouts
    # scheduled" while sharing not one word with it, so it scores zero here
    # even though the embedding model ranked its chunk first.
    #
    # Where lexical scoring finds little, fall back to the top-ranked chunk's
    # opening sentences — the retriever already judged that chunk the most
    # relevant thing in the corpus, and that judgement is better than none.
    # Only top up an answer that already has some lexical basis. When nothing
    # matched at all, the query shares no vocabulary with anything retrieved,
    # and manufacturing an answer from the top chunk would turn "I cannot
    # answer this" into a confident quotation — and would make the critic
    # report it as grounded.
    if 0 < len(chosen) < 2 and contexts:
        best = min(contexts, key=lambda c: c.get("distance", 0) or 0)
        citation = _citation(best.get("metadata"))
        for sentence in _sentences(best.get("document") or ""):
            if len(chosen) >= 3:
                break
            if sentence in seen_sentences:
                continue
            chosen.append((sentence, citation))
            seen_sentences.add(sentence)

    if not chosen:
        # Retrieval returned chunks, but none share vocabulary with the query.
        #
        # The nearest source is named in prose rather than in the [file pN]
        # citation format on purpose: that format is exactly what the critic
        # node greps for, and emitting it here would make the critic report
        # grounded=True for a message whose whole content is "I cannot answer
        # this". A non-answer must fail the grounding check.
        meta = contexts[0].get("metadata") or {}
        filename = meta.get("filename") or meta.get("source") or "unknown"
        page = meta.get("page", "?")
        return (
            "The retrieved documents do not contain wording that matches this "
            "question closely enough to quote. The nearest source was "
            f"{filename} (page {page}). Rephrasing with terms used in the "
            "documentation may help."
        )

    lines = [
        "**Answering from the retrieved documents.** Every sentence below is "
        "quoted verbatim from a source, with its citation:",
        "",
    ]
    for index, (sentence, citation) in enumerate(chosen, start=1):
        lines.append(f"{index}. {sentence} {citation}")

    lines += [
        "",
        "_Composed without a language model: sentences are selected from the "
        "retrieved text, not generated. Set `GROQ_API_KEY` for a synthesised "
        "answer._",
    ]
    return "\n".join(lines)
