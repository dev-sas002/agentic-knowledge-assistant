from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text per page with basic cleanup. Returns a list of {page, text}."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            # A page that will not extract is skipped, but silently skipping it
            # makes a half-indexed document look like a complete one.
            logger.warning("Could not extract text from %s page %d", pdf_path.name, i + 1)
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        pages.append({"page": i + 1, "text": text})
    return pages


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple character-based splitter with overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    # `start` advances by (chunk_size - overlap); an overlap at or above the
    # chunk size makes that zero or negative and the loop never terminates.
    # Reachable straight from the CLI: --chunk-size 100 --overlap 150.
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must be between 0 and chunk_size - 1 (got overlap={overlap}, chunk_size={chunk_size})"
        )
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pdf(pdf_path: Path, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Yield chunk dicts with metadata for a single PDF."""
    results: list[dict] = []
    pages = extract_text_from_pdf(pdf_path)
    for p in pages:
        page_no = p["page"]
        text = p["text"]
        if not text:
            continue
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            results.append(
                {
                    "id": f"{pdf_path.stem}-p{page_no}-c{idx}",
                    "text": ch,
                    "metadata": {
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": page_no,
                        "chunk": idx,
                        "title": pdf_path.stem.replace("_", " "),
                    },
                }
            )
    return results


def load_pdfs(source_dir: Path) -> list[Path]:
    # Case-insensitive: `glob("*.pdf")` silently ignored a user-supplied
    # REPORT.PDF dropped into data/raw, which looks like a broken indexer.
    return sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    ) if source_dir.is_dir() else []


if __name__ == "__main__":
    from pprint import pprint

    sample_dir = Path("data/raw")
    pdfs = load_pdfs(sample_dir)
    print(f"Found {len(pdfs)} PDFs in {sample_dir}")
    if pdfs:
        chunks = chunk_pdf(pdfs[0])
        print(f"First PDF: {pdfs[0].name}, chunks: {len(chunks)}")
        pprint(chunks[:2])
