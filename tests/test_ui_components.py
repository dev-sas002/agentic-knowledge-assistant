"""
The console's HTML fragments.

Retrieved passages come from staff-uploaded PDFs and are rendered with
``unsafe_allow_html``. These tests pin that markup in a document, a query or
an answer arrives on screen as text, while the console's own markup (citation
chips, pills) still renders.
"""

from __future__ import annotations

from apps.ui.components import highlight_citations, pill, replay, request_card, source_card

PAYLOAD = "<img src=x onerror=alert(1)>"


class TestEscaping:
    def test_a_passage_cannot_inject_markup(self):
        card = source_card(1, "menu.pdf p2", pill("cited", "#0f9d58"), "80% similar", PAYLOAD)
        assert "<img" not in card
        assert "&lt;img src=x onerror=alert(1)&gt;" in card

    def test_a_source_name_is_escaped(self):
        card = source_card(1, "<b>evil</b>.pdf p1", "", "n/a", "body")
        assert "<b>evil</b>" not in card

    def test_an_answer_is_escaped_but_keeps_its_citation_chips(self):
        html = highlight_citations(f"Payouts run on Tuesdays {PAYLOAD} [finance.pdf p7]")
        assert "<img" not in html
        assert "<span class='cite'>finance.pdf p7</span>" in html

    def test_apostrophes_in_answers_stay_readable(self):
        assert highlight_citations("The partner's statement") == "The partner's statement"

    def test_the_request_card_escapes_the_operator_query(self):
        card = request_card(
            status_pill="",
            risk_pill="",
            request_id="abc123",
            summary="Pause the menu item '<script>'",
            requested_by="demo_user",
            original_query=f"pause '{PAYLOAD}'",
            destination="Will be sent to",
            endpoint="/pause_item",
            colour="#e8a33d",
        )
        assert "<script>" not in card
        assert "<img" not in card
        assert "<code>/pause_item</code>" in card

    def test_pill_text_is_escaped(self):
        assert "<i>" not in pill("<i>x</i>", "#000000")


class TestReplay:
    def test_replay_reassembles_the_original_text(self):
        text = "A grounded answer [finance.pdf p7]. " * 5
        assert "".join(replay(text, chunk_size=7, delay=0)) == text
