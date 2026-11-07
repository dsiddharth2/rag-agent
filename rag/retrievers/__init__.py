from rag.index import Index
from rag.retrievers.hybrid import HybridRetriever
from rag.retrievers.naive import NaiveRetriever

METHODS = ("naive", "hybrid")


def make_retriever(
    method: str,
    index: Index,
    encoder,
    reranker=None,
):
    if method == "naive":
        return NaiveRetriever(index, encoder)
    if method == "hybrid":
        if reranker is None:
            raise ValueError("the hybrid method needs a reranker")
        return HybridRetriever(index, encoder, reranker)
    raise ValueError(f"unknown method {method!r}")
