"""
Build the Chroma index from the PDFs in ``RAW_DOCS_DIR``.

Embedding dominates the runtime, so two things matter here and nothing else
really does.

**Chunks are embedded through the cache.** A rebuild after adding one document
used to re-embed the entire corpus, because the indexer had no memory. Now the
cache is keyed on (model, chunk text): unchanged chunks are a lookup, and only
genuinely new text reaches the model. Re-running the build on an unchanged
corpus is close to instant, which is what makes "just rebuild it" a reasonable
thing to tell someone.

**Embedding goes through the provider seam**, not directly to
``SentenceTransformer``. That is what lets the eval harness and CI index a
corpus with the dependency-free ``hashing`` embedder, and what would let a
hosted embedding API be dropped in without touching this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Robust import so this file works both as a module and as a script.
try:  # `python -m ingestion.build_index`
    if __package__:
        from .chunking import chunk_pdf, load_pdfs
    else:
        raise ImportError
except ImportError:  # `python ingestion/build_index.py`
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.append(str(_root))
    from ingestion.chunking import chunk_pdf, load_pdfs  # type: ignore[no-redef]

from agents import config
from agents.embedding_cache import build_embedding_cache, embed_cached
from agents.providers import resolve_embedder


def collect_chunks(pdf_paths: list[Path], chunk_size: int, overlap: int) -> tuple[list[dict], list[str]]:
    """Chunk every PDF, skipping (and naming) the ones that will not read."""
    chunks: list[dict] = []
    failed: list[str] = []
    for path in tqdm(pdf_paths, desc="PDFs"):
        try:
            chunks.extend(chunk_pdf(path, chunk_size=chunk_size, overlap=overlap))
        except Exception as exc:
            print(f"  ! Skipping {path.name}: {type(exc).__name__}: {exc}")
            failed.append(path.name)
    return chunks, failed


def build_index(
    source_dir: Path,
    persist_dir: Path,
    model_name: str | None = None,
    collection_name: str = config.CHROMA_COLLECTION,
    rebuild: bool = False,
    batch_size: int = 64,
    chunk_size: int = 900,
    overlap: int = 150,
    cache_path: str | None = None,
) -> int:
    """Index ``source_dir`` into ``persist_dir``. Returns the record count."""
    import chromadb

    source_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = load_pdfs(source_dir)
    if not pdf_paths:
        print(f"No PDFs found in {source_dir}. Add files or run scripts/generate_pdfs.py")
        return 0

    print(f"Chunking {len(pdf_paths)} PDFs ...")
    all_chunks, failed = collect_chunks(pdf_paths, chunk_size, overlap)
    if failed:
        print(f"Skipped {len(failed)} unreadable PDF(s): {', '.join(failed)}")
    if not all_chunks:
        print("No text chunks extracted; check your PDFs.")
        return 0
    print(f"Total chunks: {len(all_chunks)}")

    embedder = resolve_embedder()
    if model_name:
        embedder.model_name = model_name  # type: ignore[attr-defined]
    cache = build_embedding_cache(config.EMBEDDING_CACHE_PATH if cache_path is None else cache_path)

    print(f"Initialising Chroma persistent client at {persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))
    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        collection = client.create_collection(collection_name)
    else:
        collection = client.get_or_create_collection(collection_name)

    print(f"Indexing into collection: {collection_name}")
    documents = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    progress = tqdm(total=len(documents), desc="Embedding + writing")
    for start in range(0, len(documents), batch_size):
        stop = min(start + batch_size, len(documents))
        batch = documents[start:stop]
        # Cached: a rebuild after a one-document change embeds only that
        # document's chunks, not the whole corpus.
        embeddings = embed_cached(embedder, batch, cache=cache)
        collection.upsert(
            documents=batch,
            ids=ids[start:stop],
            metadatas=metadatas[start:stop],
            embeddings=embeddings,
        )
        progress.update(len(batch))
    progress.close()

    try:
        count = collection.count()
    except Exception:
        count = len(documents)
    print(f"Done. Collection '{collection_name}' now has {count} records.")
    return int(count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Chroma index from PDFs.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the collection first")
    parser.add_argument("--collection", default=config.CHROMA_COLLECTION, help="Chroma collection name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/write batch size")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap (characters)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the embedding cache")
    args = parser.parse_args(argv)

    build_index(
        source_dir=Path(config.RAW_DOCS_DIR),
        persist_dir=Path(config.CHROMA_DB_DIR),
        model_name=config.EMBEDDING_MODEL,
        collection_name=args.collection,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        cache_path="" if args.no_cache else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
