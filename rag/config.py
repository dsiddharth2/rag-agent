from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
DATA_DIR = REPO_ROOT / "RAG_Dataset" / "data" / "processed"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QRELS_PATH = DATA_DIR / "qrels.jsonl"
INDEX_DIR = _PKG_DIR / "indexes"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 32

CHUNK_TOKENS = 200
CHUNK_OVERLAP = 40

RETRIEVE_DEPTH = 50
CONTEXT_K = 5
RRF_K = 60
