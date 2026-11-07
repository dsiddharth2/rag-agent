import argparse
import json
import sys
from pathlib import Path

from rag import config
from rag.io_utils import read_jsonl
from rag.retrievers.base import doc_ranking


def load_qrels(path: Path | str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        relevance = row.get("relevance", 1)
        if relevance <= 0:
            continue
        qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def gold_rank(ranked_doc_ids: list[str], gold: set[str]) -> int | None:
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold:
            return i
    return None


def compute_metrics(results, qrels, ks=(1, 3, 5, 10)):
    method = "naive"
    ranks: list[int | None] = []
    for row in results:
        query_id = row.get("query_id")
        if query_id not in qrels:
            continue
        if row.get("method"):
            method = row["method"]
        doc_ids = [c["doc_id"] for c in row["chunks"]]
        ranked = doc_ranking(doc_ids)
        ranks.append(gold_rank(ranked, qrels[query_id]))

    n = len(ranks)
    out: dict = {"method": method, "n": n}
    if n == 0:
        for k in ks:
            out[f"hit@{k}"] = 0.0
        out["mrr"] = 0.0
        return out

    for k in ks:
        hit = sum(1 for r in ranks if r is not None and r <= k) / n
        out[f"hit@{k}"] = round(hit, 4)

    mrr = sum(1 / r if r is not None else 0.0 for r in ranks) / n
    out["mrr"] = round(mrr, 4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute retrieval Hit@k and MRR.")
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL")
    parser.add_argument(
        "--qrels",
        type=Path,
        default=config.QRELS_PATH,
        help="Qrels JSONL",
    )
    args = parser.parse_args(argv)

    qrels = load_qrels(args.qrels)
    results = list(read_jsonl(args.results))
    metrics = compute_metrics(results, qrels)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
