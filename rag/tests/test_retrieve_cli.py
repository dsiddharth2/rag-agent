import json
from pathlib import Path

import pytest

from rag import config
from rag.retrieve import main
from rag.retrievers import make_retriever


@pytest.fixture
def loader(index, fake_encoder, fake_reranker):
    def _load(method: str, index_dir):
        reranker = fake_reranker if method == "hybrid" else None
        retriever = make_retriever(method, index, fake_encoder, reranker=reranker)
        return retriever, index

    return _load


def test_single_query_naive(loader, capsys):
    code = main(
        ["--method", "naive", "--query", "hvac schedule", "--depth", "2"],
        retriever_loader=loader,
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["chunks"]) == 2
    assert out["chunks"][0]["doc_id"] == "d3"
    assert out["method"] == "naive"


def test_query_file_hybrid(loader, tmp_path, capsys, index):
    qpath = tmp_path / "q.txt"
    qpath.write_text("E370 elevator door sensor", encoding="utf-8")
    code = main(
        [
            "--method",
            "hybrid",
            "--query-file",
            str(qpath),
            "--depth",
            "3",
        ],
        retriever_loader=loader,
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["chunks"][0]["doc_id"] == "d4"


def test_batch_naive(loader, tmp_path, capsys, index):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query_id":"q1","text":"hvac schedule"}\n'
        '{"query_id":"q2","text":"door"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "results.jsonl"
    code = main(
        [
            "--method",
            "naive",
            "--queries-file",
            str(queries),
            "--out",
            str(out_path),
            "--depth",
            "3",
        ],
        retriever_loader=loader,
    )
    assert code == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [r["query_id"] for r in rows] == ["q1", "q2"]
    for row in rows:
        assert row["method"] == "naive"
        assert len(row["chunks"]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["queries"] == 2
    assert summary["rerank_model"] is None
    assert summary["embed_model"] == config.EMBED_MODEL
    assert summary["out"] == str(out_path)


def test_queries_file_without_out_raises_system_exit():
    with pytest.raises(SystemExit) as exc:
        main(["--method", "naive", "--queries-file", "q.jsonl"])
    assert exc.value.code == 2


def test_missing_index_dir_exit_2(capsys, tmp_path):
    missing = tmp_path / "missing"
    code = main(
        ["--method", "naive", "--query", "x", "--index-dir", str(missing)],
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "build_index" in err
