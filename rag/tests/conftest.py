import pytest

from rag.tests.fakes import CORPUS, FakeEncoder, FakeReranker


@pytest.fixture
def corpus():
    return CORPUS


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def fake_reranker():
    return FakeReranker()


@pytest.fixture
def index(tmp_path, corpus, fake_encoder):
    from rag.index import build_index, load_index
    from rag.tests.fakes import whitespace_spans

    out_dir = tmp_path / "index"
    build_index(corpus, whitespace_spans, fake_encoder, out_dir, "abc")
    return load_index(out_dir)
