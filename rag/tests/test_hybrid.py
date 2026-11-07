import pytest

from rag import config
from rag.retrievers import make_retriever
from rag.retrievers.base import bm25_search
from rag.retrievers.hybrid import HybridRetriever, rrf_fuse


def test_rrf_fuse_overlapping_rankings():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids == ["a", "c", "b"]
    by_id = dict(fused)
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["b"] == pytest.approx(1 / 62)


def test_rrf_fuse_disjoint_tie_breaks_by_first_seen():
    fused = rrf_fuse([["x"], ["y"]], k=config.RRF_K)
    ids = [cid for cid, _ in fused]
    assert ids == ["x", "y"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_bm25_search_exact_code(index):
    hits = bm25_search(index, "E307", 10)
    assert len(hits) == 1
    assert hits[0].chunk.doc_id == "d1"


def test_hybrid_retrieve_e370(index, fake_encoder, fake_reranker):
    retriever = HybridRetriever(index, fake_encoder, fake_reranker, pool=50)
    hits = retriever.retrieve("E370 elevator door sensor", depth=3)
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "d4"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_make_retriever_hybrid(index, fake_encoder, fake_reranker):
    retriever = make_retriever("hybrid", index, fake_encoder, reranker=fake_reranker)
    assert retriever.name == "hybrid"


def test_make_retriever_hybrid_without_reranker_raises(index, fake_encoder):
    with pytest.raises(ValueError, match="reranker"):
        make_retriever("hybrid", index, fake_encoder, reranker=None)
