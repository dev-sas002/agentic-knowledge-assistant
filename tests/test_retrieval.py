"""
Tests for the retrieval layer.

Chroma's response shape is the source of most of these: it returns nested
rows, omits fields that were not requested, and can return rows of differing
length. Every one of those used to be a crash or a silent truncation.
"""

from __future__ import annotations

import pytest

from agents import config
from agents.retrieval import ChromaRetriever, Passage, _first_row, clamp_top_k
from tests.fakes import FakeCollection, FakeEmbedder


def retriever(result=None, error=None):
    return ChromaRetriever(
        embedder=FakeEmbedder(),
        collection=FakeCollection(result=result, error=error),
        use_cache=False,
    )


class TestFirstRow:
    def test_a_missing_key_yields_an_empty_row(self):
        assert _first_row({}, "documents") == []

    def test_an_explicit_none_yields_an_empty_row(self):
        # Chroma returns None for fields that were not included.
        assert _first_row({"distances": None}, "distances") == []

    def test_a_none_inner_row_yields_an_empty_row(self):
        assert _first_row({"documents": [None]}, "documents") == []

    def test_the_first_row_is_returned(self):
        assert _first_row({"documents": [["a", "b"], ["c"]]}, "documents") == ["a", "b"]


class TestClampTopK:
    def test_a_sane_value_is_unchanged(self):
        assert clamp_top_k(6) == 6

    def test_an_unbounded_value_is_capped(self):
        assert clamp_top_k(9999) == config.MAX_TOP_K

    @pytest.mark.parametrize("bad", [0, -3, None, "lots", 1.5])
    def test_nonsense_still_retrieves_something(self, bad):
        assert clamp_top_k(bad) >= 1


class TestRetrieve:
    def test_one_passage_per_hit(self):
        result = retriever().retrieve("payouts", top_k=4)
        assert len(result.passages) == 2
        assert result.passages[0].metadata["filename"] == "finance.pdf"

    def test_the_query_is_embedded_not_passed_as_text(self):
        instance = retriever()
        instance.retrieve("payouts")
        assert instance.embedder.encoded == [["payouts"]]
        assert "query_embeddings" in instance.collection.queries[0]

    def test_top_k_reaches_the_collection(self):
        instance = retriever()
        instance.retrieve("q", top_k=6)
        assert instance.collection.queries[0]["n_results"] == 6

    def test_top_k_is_clamped_before_the_collection_sees_it(self):
        instance = retriever()
        instance.retrieve("q", top_k=9999)
        assert instance.collection.queries[0]["n_results"] == config.MAX_TOP_K

    def test_embeddings_are_not_requested_back(self):
        # Including "embeddings" pulls a float vector per hit across the wire
        # for nothing.
        instance = retriever()
        instance.retrieve("q")
        assert "embeddings" not in instance.collection.queries[0]["include"]

    def test_a_failing_index_is_reported_not_raised(self):
        result = retriever(error=RuntimeError("no such collection")).retrieve("q")
        assert result.passages == []
        assert result.error == "RuntimeError"

    def test_null_metadata_is_normalised(self):
        result = retriever(
            result={"documents": [["text"]], "metadatas": [[None]], "distances": [[0.5]]}
        ).retrieve("q")
        assert result.passages[0].metadata == {}

    def test_an_empty_index_produces_no_passages_and_no_error(self):
        result = retriever(result={"documents": [[]], "metadatas": [[]], "distances": [[]]}).retrieve("q")
        assert result.passages == [] and result.error is None

    def test_a_short_distances_row_does_not_drop_passages(self):
        # zip() would silently truncate; a missing distance should show as
        # "n/a", not lose the passage.
        result = retriever(
            result={"documents": [["a", "b"]], "metadatas": [[{}, {}]], "distances": [[0.1]]}
        ).retrieve("q")
        assert len(result.passages) == 2
        assert result.passages[1].distance is None

    def test_latency_is_measured(self):
        result = retriever().retrieve("q")
        assert result.total_ms >= 0


class TestPassage:
    def test_similarity_inverts_distance(self):
        assert Passage("d", {}, 0.0).similarity == 1.0
        assert Passage("d", {}, 2.0).similarity == 0.0

    def test_similarity_is_bounded(self):
        assert Passage("d", {}, 9.0).similarity == 0.0

    def test_a_missing_distance_has_no_similarity(self):
        assert Passage("d", {}, None).similarity is None

    def test_the_source_is_the_citation_key(self):
        assert Passage("d", {"filename": "a.pdf", "page": 3}, 0.1).source == "a.pdf p3"

    def test_the_dict_shape_is_what_downstream_consumes(self):
        data = Passage("d", {"filename": "a.pdf", "page": 3}, 0.1).to_dict()
        assert set(data) == {"document", "metadata", "distance", "similarity", "source"}


class TestCaching:
    def test_the_cache_is_consulted_when_enabled(self):
        from agents import embedding_cache

        calls = []

        class CountingCache:
            def get_many(self, model, texts):
                calls.append(("get", list(texts)))
                return {}

            def put_many(self, model, pairs):
                calls.append(("put", [t for t, _ in pairs]))

        embedding_cache.set_embedding_cache(CountingCache())
        ChromaRetriever(embedder=FakeEmbedder(), collection=FakeCollection(), use_cache=True).retrieve("payouts")
        assert ("get", ["payouts"]) in calls

    def test_a_repeated_query_skips_the_model(self):
        import pathlib
        import tempfile

        from agents.embedding_cache import SqliteEmbeddingCache, embed_cached

        with tempfile.TemporaryDirectory() as tmp:
            cache = SqliteEmbeddingCache(pathlib.Path(tmp) / "c.sqlite3")
            embedder = FakeEmbedder()
            embed_cached(embedder, ["payouts"], cache=cache)
            embed_cached(embedder, ["payouts"], cache=cache)
            assert len(embedder.encoded) == 1, "the second call must not reach the model"
