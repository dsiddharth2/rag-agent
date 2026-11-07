from collections.abc import Callable
from dataclasses import dataclass

from rag import config

TokenSpans = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def window_starts(n_tokens: int, size: int, overlap: int) -> list[int]:
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if n_tokens == 0:
        return []
    step = size - overlap
    starts = [0]
    while starts[-1] + size < n_tokens:
        starts.append(starts[-1] + step)
    return starts


def chunk_document(
    doc_id: str,
    text: str,
    token_spans: TokenSpans,
    size: int = config.CHUNK_TOKENS,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    spans = token_spans(text)
    chunks: list[Chunk] = []
    for n, start in enumerate(window_starts(len(spans), size, overlap)):
        end = min(start + size, len(spans))
        chunks.append(
            Chunk(
                f"{doc_id}#{n}",
                doc_id,
                text[spans[start][0] : spans[end - 1][1]],
            )
        )
    return chunks


def hf_token_spans(tokenizer) -> TokenSpans:
    def token_spans(text: str) -> list[tuple[int, int]]:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        return [
            (start, end)
            for start, end in encoded["offset_mapping"]
            if end > start
        ]

    return token_spans
