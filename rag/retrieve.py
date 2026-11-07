import argparse
import json
import sys
from pathlib import Path

from rag import config
from rag.index import StaleIndexError, load_index
from rag.io_utils import read_jsonl, write_jsonl
from rag.models import load_embedder, load_reranker
from rag.retrievers import make_retriever
from rag.retrievers.base import to_result


def load_retriever(method: str, index_dir: Path | str):
    index = load_index(index_dir)
    encoder = load_embedder()
    reranker = load_reranker() if method == "hybrid" else None
    retriever = make_retriever(method, index, encoder, reranker=reranker)
    return retriever, index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval for one or many queries.")
    parser.add_argument("--method", choices=("naive", "hybrid"), required=True)
    parser.add_argument("--query", help="Single query text")
    parser.add_argument("--query-file", type=Path, help="UTF-8 file containing one query")
    parser.add_argument("--queries-file", type=Path, help="JSONL of {query_id, text} rows")
    parser.add_argument("--out", type=Path, help="Batch output JSONL path")
    parser.add_argument("--depth", type=int, default=config.RETRIEVE_DEPTH)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.INDEX_DIR,
        help="Directory with index artifacts",
    )
    return parser


def _rerank_model(method: str) -> str | None:
    if method == "hybrid":
        return config.RERANK_MODEL
    return None


def main(argv=None, retriever_loader=load_retriever) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.queries_file is not None:
        if args.out is None:
            parser.error("--queries-file requires --out")
        if args.query is not None or args.query_file is not None:
            parser.error("use only one of --query, --query-file, or --queries-file")
    elif args.query_file is not None:
        if args.query is not None:
            parser.error("use only one of --query, --query-file, or --queries-file")
    elif args.query is None:
        parser.error("one of --query, --query-file, or --queries-file is required")

    try:
        retriever, index = retriever_loader(args.method, args.index_dir)
    except StaleIndexError as exc:
        print(exc, file=sys.stderr)
        return 2

    embed_model = index.manifest["embed_model"]
    rerank_model = _rerank_model(args.method)

    if args.queries_file is not None:
        rows = []
        for row in read_jsonl(args.queries_file):
            query_id = row["query_id"]
            text = row["text"]
            scored = retriever.retrieve(text, depth=args.depth)
            result = to_result(args.method, text, scored)
            rows.append({"query_id": query_id, **result})
        write_jsonl(args.out, rows)
        summary = {
            "method": args.method,
            "queries": len(rows),
            "out": str(args.out),
            "depth": args.depth,
            "embed_model": embed_model,
            "rerank_model": rerank_model,
        }
        print(json.dumps(summary))
        return 0

    if args.query_file is not None:
        query = args.query_file.read_text(encoding="utf-8")
    else:
        query = args.query

    scored = retriever.retrieve(query, depth=args.depth)
    print(json.dumps(to_result(args.method, query, scored)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
