import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag import config
from rag.chunking import Chunk, TokenSpans, chunk_document
from rag.io_utils import read_jsonl, write_jsonl
from rag.lexical import bm25_tokenize
from rag.models import embed


class StaleIndexError(RuntimeError):
    """Raised when the on-disk index is missing or out of date."""


_REBUILD = "run: python -m rag.build_index"


def _stale(message: str) -> StaleIndexError:
    return StaleIndexError(f"{message}; {_REBUILD}")


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25Okapi
    manifest: dict


def index_settings() -> dict:
    return {
        "embed_model": config.EMBED_MODEL,
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def build_index(
    docs,
    token_spans: TokenSpans,
    encoder,
    out_dir: Path | str,
    corpus_sha1: str,
    settings: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = index_settings()

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc["doc_id"], doc["text"], token_spans)
        )

    if not chunks:
        raise ValueError("no chunks produced from corpus")

    chunk_rows = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text}
        for c in chunks
    ]
    write_jsonl(out_dir / "chunks.jsonl", chunk_rows)

    texts = [c.text for c in chunks]
    embeddings = embed(encoder, texts)
    np.save(out_dir / "embeddings.npy", embeddings)

    tokenized = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    manifest = {
        **settings,
        "corpus_sha1": corpus_sha1,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def load_index(index_dir: Path | str, expected: dict | None = None) -> Index:
    index_dir = Path(index_dir)
    manifest_path = index_dir / "manifest.json"

    if not manifest_path.is_file():
        raise _stale("index manifest missing (run build_index)")

    if expected is None:
        expected = index_settings()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for key, want in expected.items():
        have = manifest.get(key)
        if have != want:
            raise _stale(f"index {key}={have!r} does not match expected {want!r}")

    chunks: list[Chunk] = []
    for row in read_jsonl(index_dir / "chunks.jsonl"):
        chunks.append(
            Chunk(row["chunk_id"], row["doc_id"], row["text"])
        )

    embeddings = np.load(index_dir / "embeddings.npy")
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    n_chunks = manifest["n_chunks"]
    if len(chunks) != n_chunks:
        raise _stale(
            f"chunks.jsonl has {len(chunks)} rows but manifest n_chunks={n_chunks}"
        )
    if embeddings.shape[0] != n_chunks:
        raise _stale(
            f"embeddings rows {embeddings.shape[0]} != manifest n_chunks={n_chunks}"
        )
    if len(bm25.doc_freqs) != n_chunks:
        raise _stale(
            f"bm25 corpus size {len(bm25.doc_freqs)} != manifest n_chunks={n_chunks}"
        )

    return Index(chunks, embeddings, bm25, manifest)
