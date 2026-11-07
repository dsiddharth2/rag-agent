import pytest

from rag.chunking import chunk_document, window_starts
from rag.tests.fakes import whitespace_spans


def test_window_starts_500_tokens():
    assert window_starts(500, 200, 40) == [0, 160, 320]


def test_window_starts_fits_single_window():
    assert window_starts(10, 200, 40) == [0]
    assert window_starts(200, 200, 40) == [0]


def test_window_starts_zero_tokens():
    assert window_starts(0, 200, 40) == []


def test_window_starts_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        window_starts(10, 5, 5)


def test_chunk_document_sliding_windows():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_document("swg1", text, whitespace_spans)

    assert [c.chunk_id for c in chunks] == ["swg1#0", "swg1#1", "swg1#2"]
    assert chunks[0].text == " ".join(f"w{i}" for i in range(200))
    assert chunks[1].text == " ".join(f"w{i}" for i in range(160, 360))
    assert chunks[2].text == " ".join(f"w{i}" for i in range(320, 500))


def test_chunk_document_preserves_original_text_slice():
    text = "Title: IBM Fix\n\nRun STREAMTOOL now"
    chunks = chunk_document("doc1", text, whitespace_spans)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_document_blank():
    assert chunk_document("empty", "", whitespace_spans) == []
