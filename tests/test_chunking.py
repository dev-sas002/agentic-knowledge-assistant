"""
Tests for PDF chunking.

No PDF is read here: `extract_text_from_pdf` is replaced, so the tests cover
the splitting and metadata logic without depending on pypdf's rendering or on
the sample corpus having been generated.

The splitter is the part that matters. It advances by `chunk_size - overlap`,
and the CLI exposes both numbers, so a caller can ask for an overlap that
makes that step zero — at which point the loop never ends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import chunking


class TestSplitText:
    def test_a_short_text_is_a_single_chunk(self):
        assert chunking.split_text("hello", chunk_size=100, overlap=10) == ["hello"]

    def test_every_character_survives_the_split(self):
        text = "".join(str(i % 10) for i in range(1000))
        chunks = chunking.split_text(text, chunk_size=300, overlap=50)
        # Reassemble by dropping the overlap from each chunk after the first.
        rebuilt = chunks[0] + "".join(c[50:] for c in chunks[1:])
        assert rebuilt == text

    def test_chunks_overlap_by_the_requested_amount(self):
        text = "".join(str(i % 10) for i in range(500))
        chunks = chunking.split_text(text, chunk_size=200, overlap=40)
        assert chunks[0][-40:] == chunks[1][:40]

    def test_no_chunk_exceeds_the_size(self):
        text = "x" * 2500
        assert all(len(c) <= 400 for c in chunking.split_text(text, chunk_size=400, overlap=100))

    def test_zero_overlap_is_allowed(self):
        assert chunking.split_text("abcdef", chunk_size=2, overlap=0) == ["ab", "cd", "ef"]

    def test_an_empty_text_produces_no_chunks(self):
        assert chunking.split_text("", chunk_size=100, overlap=10) == []

    @pytest.mark.parametrize("overlap", [900, 1000, 2000])
    def test_an_overlap_at_or_above_the_chunk_size_is_refused(self, overlap):
        # Reachable straight from the CLI, and the symptom was a hang rather
        # than an error: `start` stops advancing and the loop never ends.
        with pytest.raises(ValueError, match="overlap"):
            chunking.split_text("x" * 5000, chunk_size=900, overlap=overlap)

    def test_a_negative_overlap_is_refused(self):
        with pytest.raises(ValueError, match="overlap"):
            chunking.split_text("x" * 100, chunk_size=50, overlap=-1)

    def test_a_nonsense_chunk_size_is_refused(self):
        with pytest.raises(ValueError, match="chunk_size"):
            chunking.split_text("x" * 100, chunk_size=0, overlap=0)


class TestChunkPdf:
    @pytest.fixture
    def two_pages(self, monkeypatch):
        monkeypatch.setattr(
            chunking,
            "extract_text_from_pdf",
            lambda path: [
                {"page": 1, "text": "Payouts are issued weekly."},
                {"page": 2, "text": "Disputes are reconciled in five days."},
            ],
        )
        return Path("/docs/Finance_Processes_Guide.pdf")

    def test_one_chunk_per_page_when_pages_are_short(self, two_pages):
        assert len(chunking.chunk_pdf(two_pages)) == 2

    def test_chunk_ids_are_unique_and_traceable(self, two_pages):
        ids = [c["id"] for c in chunking.chunk_pdf(two_pages)]
        assert ids == ["Finance_Processes_Guide-p1-c0", "Finance_Processes_Guide-p2-c0"]
        assert len(set(ids)) == len(ids)

    def test_metadata_carries_what_the_citation_needs(self, two_pages):
        meta = chunking.chunk_pdf(two_pages)[0]["metadata"]
        # agents/local_composer.py formats [filename pN] out of exactly these.
        assert meta["filename"] == "Finance_Processes_Guide.pdf"
        assert meta["page"] == 1
        assert meta["title"] == "Finance Processes Guide"

    def test_blank_pages_are_skipped(self, monkeypatch):
        monkeypatch.setattr(
            chunking,
            "extract_text_from_pdf",
            lambda path: [{"page": 1, "text": ""}, {"page": 2, "text": "Real content here."}],
        )
        chunks = chunking.chunk_pdf(Path("/docs/a.pdf"))
        assert [c["metadata"]["page"] for c in chunks] == [2]

    def test_a_long_page_becomes_several_numbered_chunks(self, monkeypatch):
        monkeypatch.setattr(
            chunking,
            "extract_text_from_pdf",
            lambda path: [{"page": 1, "text": "word " * 600}],
        )
        chunks = chunking.chunk_pdf(Path("/docs/a.pdf"), chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert [c["metadata"]["chunk"] for c in chunks] == list(range(len(chunks)))


class TestLoadPdfs:
    def test_only_pdfs_are_picked_up_and_in_a_stable_order(self, tmp_path):
        for name in ["b.pdf", "a.pdf", "notes.txt"]:
            (tmp_path / name).write_text("x")
        (tmp_path / "subdir.pdf").mkdir()
        found = [p.name for p in chunking.load_pdfs(tmp_path)]
        assert found == ["a.pdf", "b.pdf"]

    def test_an_uppercase_extension_is_still_a_pdf(self, tmp_path):
        # Someone dropping REPORT.PDF into data/raw and getting "No PDFs
        # found" reads as a broken indexer.
        (tmp_path / "REPORT.PDF").write_text("x")
        assert [p.name for p in chunking.load_pdfs(tmp_path)] == ["REPORT.PDF"]

    def test_an_empty_directory_is_not_an_error(self, tmp_path):
        assert chunking.load_pdfs(tmp_path) == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert chunking.load_pdfs(tmp_path / "nope") == []
