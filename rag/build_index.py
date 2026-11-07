import argparse
import hashlib
import json
from pathlib import Path

from rag import config
from rag.chunking import hf_token_spans
from rag.io_utils import read_jsonl
from rag.index import build_index
from rag.models import load_embedder, load_reranker, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk index artifacts.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=config.CORPUS_PATH,
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.INDEX_DIR,
        help="Output directory for index artifacts",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)

    load_reranker()

    corpus_sha1 = hashlib.sha1(corpus_path.read_bytes()).hexdigest()
    docs = list(read_jsonl(corpus_path))

    tokenizer = load_tokenizer()
    token_spans = hf_token_spans(tokenizer)
    encoder = load_embedder()

    manifest = build_index(docs, token_spans, encoder, out_dir, corpus_sha1)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
