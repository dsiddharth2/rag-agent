from rag import config
from rag.index import Index
from rag.retrievers.base import ScoredChunk, bm25_search, dense_search


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk_id, len(first_seen))
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, index: Index, encoder, reranker, pool: int = 50):
        self._index = index
        self._encoder = encoder
        self._reranker = reranker
        self._pool = pool
        self._chunk_by_id = {c.chunk_id: c for c in index.chunks}

    def retrieve(self, query: str, depth: int = 50) -> list[ScoredChunk]:
        pool = self._pool
        dense_hits = dense_search(self._index, self._encoder, query, pool)
        bm25_hits = bm25_search(self._index, query, pool)
        dense_ids = [h.chunk.chunk_id for h in dense_hits]
        bm25_ids = [h.chunk.chunk_id for h in bm25_hits]
        fused = rrf_fuse([dense_ids, bm25_ids], k=config.RRF_K)
        fused_top = fused[:pool]
        pairs = [(query, self._chunk_by_id[cid].text) for cid, _ in fused_top]
        rerank_scores = self._reranker.predict(
            pairs,
            batch_size=config.RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        scored = [
            ScoredChunk(self._chunk_by_id[cid], float(score))
            for (cid, _), score in zip(fused_top, rerank_scores, strict=True)
        ]
        scored.sort(key=lambda h: -h.score)
        return scored[:depth]
