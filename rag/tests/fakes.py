import re
from typing import Any

import numpy as np

CORPUS = [
    {"doc_id": "d1", "text": "elevator motor overheating fault E307 on tower B"},
    {"doc_id": "d2", "text": "door sensor stuck fault reported by technicians"},
    {"doc_id": "d3", "text": "hvac night schedule setback configuration guide"},
    {"doc_id": "d4", "text": "error E370 indicates elevator door sensor misalignment"},
]
VOCAB = ["elevator", "door", "hvac", "e307", "e370", "motor", "sensor", "schedule"]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def whitespace_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class FakeEncoder:
    def encode(self, sentences, **kwargs: Any) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        dim = len(VOCAB)
        vectors = np.zeros((len(sentences), dim), dtype=np.float64)
        for row, sentence in enumerate(sentences):
            for word in _tokenize(sentence):
                if word in VOCAB:
                    vectors[row, VOCAB.index(word)] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class FakeReranker:
    def predict(self, pairs, **kwargs: Any) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            query_words = set(_tokenize(query))
            doc_words = set(_tokenize(doc))
            scores.append(float(len(query_words & doc_words)))
        return scores
