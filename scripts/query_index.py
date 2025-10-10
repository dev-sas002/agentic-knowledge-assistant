"""
Inspect retrieval on its own, without the UI or the graph.

The first question asked of a bad RAG answer is "was it retrieval or
composition?". This answers that half directly.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.retrieval import get_retriever  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Chroma index and view citations.")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=4, help="Number of passages")
    args = parser.parse_args(argv)

    result = get_retriever().retrieve(args.q, top_k=args.k)

    if result.error:
        print(f"Retrieval failed: {result.error}")
        return 1
    if not result.passages:
        print("No matches. Has the index been built (python -m ingestion.build_index)?")
        return 1

    print(f"\nTop {len(result.passages)} passages "
          f"(embed {result.embed_ms:.0f}ms, query {result.query_ms:.0f}ms):\n")
    for position, passage in enumerate(result.passages, start=1):
        title = (passage.metadata or {}).get("title", "")
        similarity = f"{passage.similarity:.1%}" if passage.similarity is not None else "n/a"
        print(f"[{position}] {passage.source} — {title} | similarity={similarity}")
        snippet = (passage.document or "").replace("\n", " ")
        print(f"    {snippet[:300]}{'...' if len(snippet) > 300 else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
