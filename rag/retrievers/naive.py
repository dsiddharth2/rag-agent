from rag.index import Index
from rag.retrievers.base import ScoredChunk, dense_search


class NaiveRetriever:
    name = "naive"

    def __init__(self, index: Index, encoder) -> None:
        self._index = index
        self._encoder = encoder

    def retrieve(self, query: str, depth: int = 50) -> list[ScoredChunk]:
        return dense_search(self._index, self._encoder, query, depth)
