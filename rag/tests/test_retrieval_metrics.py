import json
from pathlib import Path

from rag.io_utils import write_jsonl
from rag.retrieval_metrics import compute_metrics, gold_rank, load_qrels, main


def _chunk(doc_id: str, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": f"{doc_id}#0",
        "doc_id": doc_id,
        "score": 0.5,
        "text": "t",
    }


def _result(query_id: str, doc_ids: list[str], method: str = "naive") -> dict:
    chunks = [_chunk(d, i) for i, d in enumerate(doc_ids, start=1)]
    return {
        "query_id": query_id,
        "method": method,
        "query": query_id,
        "chunks": chunks,
    }


def test_gold_rank_and_compute_metrics():
    qrels = {
        "q1": {"g1"},
        "q2": {"g2"},
        "q3": {"g3"},
    }
    results = [
        _result("q1", ["g1", "x"]),
        _result("q2", ["x", "x", "y", "g2"]),
        _result("q3", ["x", "y"]),
        _result("q4", ["z"]),
    ]

    ranked_q2 = ["x", "y", "g2"]
    assert gold_rank(ranked_q2, {"g2"}) == 3
    assert gold_rank(["x", "y"], {"g3"}) is None

    metrics = compute_metrics(results, qrels)
    assert metrics["method"] == "naive"
    assert metrics["n"] == 3
    assert metrics["hit@1"] == 0.3333
    assert metrics["hit@3"] == 0.6667
    assert metrics["hit@5"] == 0.6667
    assert metrics["hit@10"] == 0.6667
    assert metrics["mrr"] == 0.4444


def test_load_qrels(tmp_path):
    path = tmp_path / "qrels.jsonl"
    write_jsonl(
        path,
        [
            {"query_id": "q1", "doc_id": "g1", "relevance": 1},
            {"query_id": "q2", "doc_id": "g2", "relevance": 1},
        ],
    )
    assert load_qrels(path) == {"q1": {"g1"}, "q2": {"g2"}}


def test_cli_prints_metrics(tmp_path, capsys):
    results_path = tmp_path / "results.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(
        qrels_path,
        [
            {"query_id": "q1", "doc_id": "g1", "relevance": 1},
            {"query_id": "q2", "doc_id": "g2", "relevance": 1},
            {"query_id": "q3", "doc_id": "g3", "relevance": 1},
        ],
    )
    write_jsonl(
        results_path,
        [
            _result("q1", ["g1", "x"]),
            _result("q2", ["x", "x", "y", "g2"]),
            _result("q3", ["x", "y"]),
            _result("q4", ["z"]),
        ],
    )
    code = main(["--results", str(results_path), "--qrels", str(qrels_path)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 3
    assert out["hit@1"] == 0.3333
    assert out["hit@3"] == 0.6667
    assert out["mrr"] == 0.4444
