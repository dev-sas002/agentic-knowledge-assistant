"""
Tests for extractive composition.

The property that matters is that nothing is invented. A retrieval system
exists to keep answers tied to sources, so an offline mode that fabricated
prose would demonstrate exactly the failure the system is meant to prevent.
Every assertion here is about text appearing verbatim in a source, or about a
citation the critic node can recognise.
"""

from __future__ import annotations

import re

from agents.local_composer import _tokens, compose_extractive, score_sentence

# The critic node's grounding check, copied deliberately: if these drift apart,
# the offline answer stops passing the check and these tests should say so.
CITATION_RE = re.compile(r"\[[^\]]+ p\d+\]")


def ctx(text: str, filename: str = "handbook.pdf", page: int = 3) -> dict:
    return {"document": text, "metadata": {"filename": filename, "page": page}}


ONBOARDING = ctx(
    "Merchant onboarding requires a signed agreement before any menu is "
    "published. The operations team verifies bank details within two business "
    "days of submission. Menu items cannot be paused until onboarding "
    "completes.",
    filename="onboarding.pdf",
    page=2,
)

FINANCE = ctx(
    "Payouts are issued weekly on Tuesdays for the preceding week. A payout is "
    "held if the bank account fails verification. Finance reconciles disputed "
    "charges within five business days.",
    filename="finance.pdf",
    page=7,
)


class TestGrounding:
    def test_every_quoted_sentence_appears_in_a_source(self):
        answer = compose_extractive("How are payouts issued?", [FINANCE])

        source_text = FINANCE["document"]
        quoted = [
            line.split(". ", 1)[1].rsplit("[", 1)[0].strip()
            for line in answer.splitlines()
            if re.match(r"^\d+\. ", line)
        ]
        assert quoted, "no sentences were quoted"
        for sentence in quoted:
            assert sentence in source_text, f"invented text: {sentence!r}"

    def test_the_answer_carries_a_citation_the_critic_recognises(self):
        answer = compose_extractive("How are payouts issued?", [FINANCE])
        assert CITATION_RE.search(answer), "critic's grounding check would fail"

    def test_citations_name_the_source_and_page(self):
        answer = compose_extractive("How are payouts issued?", [FINANCE])
        assert "[finance.pdf p7]" in answer


class TestRelevance:
    def test_it_selects_sentences_that_match_the_question(self):
        answer = compose_extractive("When are payouts issued?", [FINANCE])
        assert "Tuesdays" in answer

    def test_it_draws_on_more_than_one_source_when_several_are_retrieved(self):
        answer = compose_extractive(
            "What happens with bank verification?", [ONBOARDING, FINANCE]
        )
        assert "[onboarding.pdf p2]" in answer
        assert "[finance.pdf p7]" in answer

    def test_a_concise_style_quotes_less(self):
        detailed = compose_extractive("bank verification menu payout", [ONBOARDING, FINANCE])
        concise = compose_extractive(
            "bank verification menu payout", [ONBOARDING, FINANCE], style="concise"
        )
        def count(text):
            return len([line for line in text.splitlines() if re.match(r"^\d+\. ", line)])

        # `<=` on its own would pass even if the style were ignored entirely.
        assert count(detailed) > count(concise)
        assert count(concise) <= 3


class TestHonestFailure:
    def test_no_contexts_says_so_rather_than_answering(self):
        answer = compose_extractive("anything at all", [])
        assert "no supporting documents" in answer.lower()
        assert not CITATION_RE.search(answer)

    def test_irrelevant_contexts_are_admitted_not_padded(self):
        # Retrieval returned something, but it shares no vocabulary with the
        # question. Quoting it anyway would be a confident non-answer.
        answer = compose_extractive(
            "quantum chromodynamics lattice gauge", [FINANCE]
        )
        assert "do not contain" in answer.lower()

    def test_it_never_claims_to_have_generated_prose(self):
        answer = compose_extractive("How are payouts issued?", [FINANCE])
        assert "without a language model" in answer.lower()


class TestScoring:
    def test_a_sentence_sharing_query_words_outranks_one_that_does_not(self):
        query = _tokens("payout schedule")
        assert score_sentence("Payouts are issued weekly.", query) > score_sentence(
            "The kitchen closes at midnight.", query
        )

    def test_length_does_not_win_on_its_own(self):
        # Normalising by sentence length stops a long paragraph outranking a
        # precise sentence purely by containing more words.
        query = _tokens("payout")
        precise = "Payouts are issued weekly."
        padded = (
            "Payouts are issued weekly and the team also handles menus, "
            "onboarding, disputes, reconciliation, reporting, and a range of "
            "other operational duties across many partner accounts."
        )
        assert score_sentence(precise, query) > score_sentence(padded, query)

    def test_stopwords_do_not_create_relevance(self):
        query = _tokens("what is the")
        assert score_sentence("This is the thing that we have.", query) == 0.0


class TestNonAnswersFailTheGroundingCheck:
    """
    The critic greps the answer for `[filename pN]`. A message whose content is
    "I cannot answer this" must not contain that pattern, or the critic reports
    a confident non-answer as grounded.
    """

    def test_an_unanswerable_query_produces_no_citation_pattern(self):
        answer = compose_extractive("quantum chromodynamics lattice gauge", [FINANCE])
        assert not CITATION_RE.search(answer)
        # The source is still named, just not in the citation format.
        assert "finance.pdf" in answer

    def test_no_contexts_produces_no_citation_pattern(self):
        assert not CITATION_RE.search(compose_extractive("anything", []))

    def test_a_real_answer_does_produce_one(self):
        assert CITATION_RE.search(compose_extractive("When are payouts issued?", [FINANCE]))


class TestStemming:
    """
    People ask about a "payout"; documents say "payouts". Without this, an
    obviously relevant sentence scores exactly zero and the composer reports
    that it found nothing to quote.
    """

    def test_a_singular_query_matches_a_plural_document(self):
        answer = compose_extractive("payout schedule", [FINANCE])
        assert "Payouts are issued weekly" in answer

    def test_plural_and_singular_produce_the_same_tokens(self):
        assert _tokens("payout") == _tokens("payouts")
        assert _tokens("policy") == _tokens("policies")

    def test_short_words_are_not_over_stemmed(self):
        # Stripping "s" from a four-letter word turns "bass" into "bas" and
        # starts matching unrelated text.
        assert _tokens("bass") == {"bass"}
        assert "gas" in _tokens("gas prices")

    def test_a_stem_is_not_truncated_into_a_different_word(self):
        # "policies" must come back as "policy", not "polic" or "poli".
        assert _tokens("policies") == {"policy"}


class TestMultipleChunksFromOneSource:
    """
    Retrieval routinely returns several chunks from the same page, which share
    a citation. A selection pass that skipped repeated citations collapsed the
    entire answer to a single sentence.
    """

    @staticmethod
    def _same_page_chunks():
        return [
            {"document": text, "metadata": {"filename": "finance.pdf", "page": 1}}
            for text in [
                "Payments are issued weekly on Fridays. Statements summarize orders and fees.",
                "Disputes must be raised within 7 days. Reconciliation aligns order data with payouts.",
                "Payout schedules can be changed by finance with approval from operations.",
            ]
        ]

    def test_it_quotes_more_than_one_sentence(self):
        answer = compose_extractive("payout schedule reconciliation", self._same_page_chunks())
        quoted = [line for line in answer.splitlines() if re.match(r"^\d+\. ", line)]
        assert len(quoted) >= 2

    def test_it_still_respects_the_sentence_cap(self):
        answer = compose_extractive(
            "payout schedule reconciliation statements disputes",
            self._same_page_chunks(),
            max_sentences=2,
        )
        quoted = [line for line in answer.splitlines() if re.match(r"^\d+\. ", line)]
        assert len(quoted) <= 2

    def test_no_sentence_is_quoted_twice(self):
        answer = compose_extractive("payout schedule", self._same_page_chunks() * 2)
        quoted = [re.sub(r"^\d+\. ", "", line) for line in answer.splitlines() if re.match(r"^\d+\. ", line)]
        assert len(quoted) == len(set(quoted))


class TestSemanticRetrievalFallback:
    """
    Retrieval is semantic; sentence selection here is lexical. A sentence can
    answer the question while sharing no words with it, so an answer with only
    one lexical match is topped up from the chunk the retriever ranked first.

    The line that matters: this tops up an answer that already has a basis. It
    must never manufacture one where nothing matched at all.
    """

    def test_a_thin_answer_is_topped_up_from_the_best_chunk(self):
        chunks = [
            {
                "document": (
                    "Payments are issued weekly on Fridays. Statements "
                    "summarize orders, fees and adjustments. Payouts follow "
                    "the same cycle."
                ),
                "metadata": {"filename": "finance.pdf", "page": 1},
                "distance": 0.2,
            }
        ]
        answer = compose_extractive("payout", chunks)
        quoted = [line for line in answer.splitlines() if re.match(r"^\d+\. ", line)]
        assert len(quoted) >= 2

    def test_nothing_matching_still_reports_that_it_cannot_answer(self):
        chunks = [{**FINANCE, "distance": 0.9}]
        answer = compose_extractive("quantum chromodynamics lattice gauge", chunks)
        assert "do not contain" in answer.lower()
        assert not CITATION_RE.search(answer)
