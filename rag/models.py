import numpy as np

from rag import config


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    tokenizer.model_max_length = 10_000_000
    return tokenizer


def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def load_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_LENGTH)


def embed(encoder, texts: list[str]) -> np.ndarray:
    vectors = encoder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
