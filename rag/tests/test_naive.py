import pytest

from rag.chunking import Chunk
from rag.retrievers import make_retriever
from rag.retrievers.base import ScoredChunk, doc_ranking, to_result
from rag.retrievers.naive import NaiveRetriever


def test_naive_retrieve_hvac_schedule(index, fake_encoder):
    retriever = NaiveRetriever(index, fake_encoder)
    hits = retriever.retrieve("hvac schedule", depth=2)
    assert len(hits) == 2
    assert hits[0].chunk.doc_id == "d3"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_naive_retrieve_door_returns_all_chunks(index, fake_encoder):
    retriever = NaiveRetriever(index, fake_encoder)
    hits = retriever.retrieve("door", depth=50)
    assert len(hits) == 4


def test_doc_ranking_deduplicates_in_order():
    assert doc_ranking(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_to_result_shape():
    scored = [
        ScoredChunk(Chunk("c1#0", "d1", "one"), 0.9),
        ScoredChunk(Chunk("c2#0", "d2", "two"), 0.5),
        ScoredChunk(Chunk("c3#0", "d3", "three"), 0.1),
    ]
    result = to_result("naive", "q", scored)
    assert result["method"] == "naive"
    assert result["query"] == "q"
    assert [c["rank"] for c in result["chunks"]] == [1, 2, 3]
    for row in result["chunks"]:
        assert set(row.keys()) == {"rank", "chunk_id", "doc_id", "score", "text"}


def test_make_retriever_naive(index, fake_encoder):
    retriever = make_retriever("naive", index, fake_encoder)
    assert retriever.name == "naive"


def test_make_retriever_unknown_method_raises(index, fake_encoder):
    with pytest.raises(ValueError, match="unknown method"):
        make_retriever("graph", index, fake_encoder)
