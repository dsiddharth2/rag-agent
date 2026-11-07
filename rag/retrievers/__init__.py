from rag.index import Index
from rag.retrievers.naive import NaiveRetriever

METHODS = ("naive",)


def make_retriever(
    method: str,
    index: Index,
    encoder,
    reranker=None,
):
    if method == "naive":
        return NaiveRetriever(index, encoder)
    raise ValueError(f"unknown method {method!r}")
