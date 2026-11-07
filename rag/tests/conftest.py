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
