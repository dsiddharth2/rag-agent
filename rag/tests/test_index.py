import numpy as np
import pytest

from rag.index import StaleIndexError, build_index, index_settings, load_index
from rag.tests.fakes import CORPUS, whitespace_spans


def test_build_load_round_trip(tmp_path, fake_encoder):
    out_dir = tmp_path / "idx"
    manifest = build_index(CORPUS, whitespace_spans, fake_encoder, out_dir, "abc")
    idx = load_index(out_dir)

    assert manifest["n_chunks"] == 4
    assert manifest["n_docs"] == 4
    assert manifest["corpus_sha1"] == "abc"
    assert [c.chunk_id for c in idx.chunks] == ["d1#0", "d2#0", "d3#0", "d4#0"]
    assert idx.embeddings.shape == (4, 8)
    assert idx.embeddings.dtype == np.float32
    norms = np.linalg.norm(idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    scores = idx.bm25.get_scores(["e307"])
    assert len(scores) == 4


def test_load_index_missing_raises_stale(tmp_path):
    with pytest.raises(StaleIndexError, match="build_index"):
        load_index(tmp_path / "missing")


def test_load_index_setting_mismatch_raises_stale(tmp_path, fake_encoder):
    out_dir = tmp_path / "idx"
    build_index(CORPUS, whitespace_spans, fake_encoder, out_dir, "abc")
    bad_expected = {**index_settings(), "chunk_tokens": 999}
    with pytest.raises(StaleIndexError, match="chunk_tokens"):
        load_index(out_dir, expected=bad_expected)


def test_build_index_blank_doc_raises_value_error(tmp_path, fake_encoder):
    with pytest.raises(ValueError):
        build_index(
            [{"doc_id": "empty", "text": ""}],
            whitespace_spans,
            fake_encoder,
            tmp_path / "idx",
            "sha",
        )
