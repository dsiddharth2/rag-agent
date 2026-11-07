from rag.lexical import bm25_tokenize


def test_bm25_tokenize_version_and_apar():
    assert bm25_tokenize("Upgrade to 4.1.1.2 fails: SQLCODE401 (APAR PI19804)") == [
        "upgrade",
        "to",
        "4.1.1.2",
        "fails",
        "sqlcode401",
        "apar",
        "pi19804",
    ]


def test_bm25_tokenize_slash_and_punctuation():
    assert bm25_tokenize("door/sensor, stuck!") == ["door", "sensor", "stuck"]
