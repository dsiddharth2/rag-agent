from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk
from rag.index import Index
from rag.lexical import bm25_tokenize
from rag.models import embed


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def top_indices(scores, depth: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    return list(order[:depth])


def dense_search(index: Index, encoder, query: str, depth: int) -> list[ScoredChunk]:
    query_vector = embed(encoder, [query])[0]
    scores = index.embeddings @ query_vector
    hits: list[ScoredChunk] = []
    for i in top_indices(scores, depth):
        hits.append(ScoredChunk(index.chunks[i], float(scores[i])))
    return hits


def bm25_search(index: Index, query: str, depth: int) -> list[ScoredChunk]:
    tokens = bm25_tokenize(query)
    scores = index.bm25.get_scores(tokens)
    hits: list[ScoredChunk] = []
    for i in top_indices(scores, depth):
        score = float(scores[i])
        if score <= 0:
            continue
        hits.append(ScoredChunk(index.chunks[i], score))
    return hits


def doc_ranking(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


def to_result(method: str, query: str, scored: list[ScoredChunk]) -> dict:
    chunks = []
    for rank, hit in enumerate(scored, start=1):
        chunks.append(
            {
                "rank": rank,
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "score": round(hit.score, 6),
                "text": hit.chunk.text,
            }
        )
    return {"method": method, "query": query, "chunks": chunks}
