"""
Tests for the embedding cache.

The cache is an optimisation, so the most important property is that it is
never load-bearing: a wrong hit must be impossible, and an unusable cache must
slow the system down rather than stop it.
"""

from __future__ import annotations

import pytest

from agents.embedding_cache import (
    NullEmbeddingCache,
    SqliteEmbeddingCache,
    build_embedding_cache,
    cache_key,
    embed_cached,
)
from tests.fakes import FakeEmbedder


@pytest.fixture
def cache(tmp_path):
    return SqliteEmbeddingCache(tmp_path / "embeddings.sqlite3")


class TestKeying:
    def test_the_same_text_and_model_key_the_same(self):
        assert cache_key("m", "payouts") == cache_key("m", "payouts")

    def test_a_different_model_keys_differently(self):
        # Otherwise switching EMBEDDING_MODEL serves vectors from the old one.
        assert cache_key("bge-small", "payouts") != cache_key("bge-large", "payouts")

    def test_a_different_text_keys_differently(self):
        assert cache_key("m", "payouts") != cache_key("m", "payout")

    def test_the_separator_prevents_a_collision(self):
        # "ab" + "c" and "a" + "bc" must not hash the same.
        assert cache_key("ab", "c") != cache_key("a", "bc")


class TestRoundTrip:
    def test_a_stored_vector_comes_back(self, cache):
        cache.put_many("m", [("payouts", [0.1, 0.2, 0.3])])
        found = cache.get_many("m", ["payouts"])
        assert found["payouts"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)

    def test_a_miss_is_simply_absent(self, cache):
        assert cache.get_many("m", ["nothing stored"]) == {}

    def test_a_vector_stored_under_one_model_is_a_miss_under_another(self, cache):
        cache.put_many("bge-small", [("payouts", [0.1])])
        assert cache.get_many("bge-large", ["payouts"]) == {}

    def test_it_survives_reopening(self, tmp_path):
        path = tmp_path / "c.sqlite3"
        SqliteEmbeddingCache(path).put_many("m", [("payouts", [0.5])])
        assert "payouts" in SqliteEmbeddingCache(path).get_many("m", ["payouts"])

    def test_a_large_batch_does_not_exceed_the_variable_limit(self, cache):
        texts = [f"chunk {n}" for n in range(1500)]
        cache.put_many("m", [(text, [float(len(text))]) for text in texts])
        assert len(cache.get_many("m", texts)) == 1500

    def test_empty_inputs_are_no_ops(self, cache):
        cache.put_many("m", [])
        assert cache.get_many("m", []) == {}


class TestEmbedCached:
    def test_only_misses_reach_the_model(self, cache):
        embedder = FakeEmbedder()
        embed_cached(embedder, ["a", "b"], cache=cache)
        embed_cached(embedder, ["a", "b", "c"], cache=cache)
        assert embedder.encoded == [["a", "b"], ["c"]]

    def test_order_is_preserved(self, cache):
        embedder = FakeEmbedder()
        cache.put_many(embedder.model_name, [("b", [9.0, 9.0, 9.0])])
        vectors = embed_cached(embedder, ["a", "b"], cache=cache)
        assert vectors[1] == pytest.approx([9.0, 9.0, 9.0], abs=1e-5)

    def test_duplicates_within_one_batch_are_encoded_once(self, cache):
        # Chunk overlap makes byte-identical neighbouring chunks common.
        embedder = FakeEmbedder()
        embed_cached(embedder, ["a", "a", "a"], cache=cache)
        assert embedder.encoded == [["a"]]

    def test_an_empty_batch_does_not_touch_the_model(self, cache):
        embedder = FakeEmbedder()
        assert embed_cached(embedder, [], cache=cache) == []
        assert embedder.encoded == []

    def test_the_null_cache_always_reaches_the_model(self):
        embedder = FakeEmbedder()
        embed_cached(embedder, ["a"], cache=NullEmbeddingCache())
        embed_cached(embedder, ["a"], cache=NullEmbeddingCache())
        assert len(embedder.encoded) == 2


class TestDegradation:
    def test_an_empty_path_disables_the_cache(self):
        assert isinstance(build_embedding_cache(""), NullEmbeddingCache)

    def test_an_unusable_path_degrades_rather_than_raising(self, tmp_path):
        # A read-only mount must slow the system down, not stop it.
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        assert isinstance(build_embedding_cache(str(blocker / "sub" / "c.sqlite3")), NullEmbeddingCache)

    def test_a_valid_path_builds_a_real_cache(self, tmp_path):
        assert isinstance(build_embedding_cache(str(tmp_path / "c.sqlite3")), SqliteEmbeddingCache)
