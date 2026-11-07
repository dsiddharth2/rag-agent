# Naive vs Hybrid + Rerank RAG with Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-11-07-naive-hybrid-rag-eval-design.md`

**Goal:** Two retrieval strategies (naive, hybrid + rerank) over the TechQA
corpus, an Apra agent kit workflow that answers questions with either one, and
an eval command that compares them on retrieval and answer quality.

**Architecture:** A Python package `rag/` builds the index and runs retrieval
and retrieval metrics as command-line tools. An Apra agent kit project `agent/`
calls those tools through Fleet `command()`, generates answers on the `doer`
member, judges them on the `reviewer` member, and writes reports.

**Tech stack:** Python 3.10+, numpy, rank-bm25, sentence-transformers, pytest.
Node ≥ 22.16, `@apralabs/create-agent-kit@0.3.0` (Apra Fleet workflow kit),
`node --test`.

**Out of scope for this plan:** spec §9 (Phase 2 failure analysis — gold-doc
rank field, gold-passage check, failure buckets, question tags, baseline diff).

## Global constraints

- Do not modify anything under `RAG_Dataset/`.
- Embedding model `sentence-transformers/all-MiniLM-L6-v2`; reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`. Both run locally.
- Chunks: 200 tokens (MiniLM tokenizer), 40-token overlap, ID `<doc_id>#<n>` from 0.
- `RETRIEVE_DEPTH = 50`, `CONTEXT_K = 5`, `RRF_K = 60`.
- Generator: `agent(prompt, { member_name: 'doer', model: 'standard' })`.
- Judge: `agent(prompt, { member_name: 'reviewer', model: 'premium' })`.
- Workflows address members only as `doer` / `reviewer`, never by real name.
- Members do not inherit the shell environment: Python is called by absolute path to the repo venv.
- Commands sent to members must work in POSIX `sh` (Linux/macOS) and in PowerShell (Windows). Paths are single-quoted, which is literal in both.
- A stop request or budget limit (`err.code` `CANCELLED` / `BUDGET_EXCEEDED`) ends the run. Every other Fleet error becomes an error row.
- Unit tests never download models or call Fleet. Python tests use fake encoders; Node tests use fake `command` / `agent`.
- Every task ends with its tests passing and one commit.

## Kit facts this plan relies on

Checked against `apra-agent-kit` and `apra-fleet` source:

- Workflow body = `export async function main(context)`; `context` gives `command`, `agent`, `log`, `args`. `args` is the object passed to `engine.executeFile(script, args)`.
- `command(cmd, { member_name, failSoft, timeout_s, timeoutMs })` returns stdout as a string and throws on a non-zero exit, unless `failSoft: true`. With `failSoft` it always returns `{ ok, output, error }`. On failure `output` is `''` and `error` holds the exit code, stdout and stderr.
- `agent(prompt, { member_name, model })` returns the reply as a string; `model` takes a tier (`cheap|standard|premium`).
- On Windows, Fleet runs member commands in PowerShell, prefixed with `Set-Location "<work folder>";`. `cd /d`, `&&` (PowerShell 5.1) and a bare quoted exe path do not work there.
- Members get a clean login environment. Only `HOME`, `USER`, `LOGNAME` and `SHELL` pass through, so `HF_HOME`, `HF_TOKEN` and proxy variables do not.
- Launchers (`main.mjs`) use `withStandaloneLease` from `workflows/standalone.mjs` for CLI runs — see the starter `workflows/hello/main.mjs`. `withStandaloneLease` starts the global `apra-fleet` binary from `PATH`.
- `engine.executeFile(script, args)` resolves `script` against the current directory, so launchers pass `path.join(here, '<body>.js')`.
- Stop requests throw `CancelledError` (`code: 'CANCELLED'`) and budget limits throw `BudgetExceededError` (`code: 'BUDGET_EXCEEDED'`), both from `agent()` / `command()`.
- Tools are registered in `mcp/registry.mjs` with `name`, `description`, zod `inputSchema`, optional `routing` (`{ description, args: { <name>: { extract: '…' } } }`), and `run()`.
- The kit's vetting lint flags `fs`, `child_process`, `os`, `crypto` and `process.env` in workflow body files (advisory). Keep those in helper modules, not in the body file.
- The kit's own `npm run eval` writes to `evals/reports/`. Our reports go to `evals/reports/rag/` to stay separate.
- Scaffolding: `npm create @apralabs/agent-kit@0.3.0 agent -- --no-install`. Without `--no-install` it runs `git init` inside our repo. With it, the global `apra-fleet` / `claude` CLI install is skipped too (Task 13 setup covers that).

## File map

```
.gitignore                                  modify: add rag/indexes/, __pycache__/, .pytest_cache/
README.md                                   modify: setup + usage
rag/
  __init__.py
  requirements.txt
  config.py                                 paths and constants
  io_utils.py                               read_jsonl / write_jsonl
  chunking.py                               Chunk, window_starts, chunk_document, hf_token_spans
  lexical.py                                bm25_tokenize
  models.py                                 load_tokenizer / load_embedder / load_reranker, embed
  index.py                                  build_index, load_index, StaleIndexError
  build_index.py                            CLI
  retrievers/__init__.py                    METHODS, make_retriever
  retrievers/base.py                        ScoredChunk, dense_search, bm25_search, doc_ranking, to_result
  retrievers/naive.py                       NaiveRetriever
  retrievers/hybrid.py                      rrf_fuse, HybridRetriever
  retrieve.py                               CLI (single / query-file / batch)
  retrieval_metrics.py                      CLI + compute_metrics
  tests/                                    fakes.py, conftest.py, test_*.py
agent/                                      generated by the kit, then:
  package.json                              modify: rag:answer, rag:eval scripts
  host.config.mjs                           modify: TechQA description and agentDescription
  mcp/registry.mjs                          modify: rag_answer tool
  workflows/rag-shared/paths.mjs            repo paths, python command builder
  workflows/rag-shared/llm-json.mjs         toText, extractJsonObject
  workflows/rag-shared/retrieval-client.mjs single / batch / metrics over command()
  workflows/rag-shared/generation.mjs       answer prompt, parse, generateAnswer
  workflows/rag-answer/rag-answer.js        body
  workflows/rag-answer/main.mjs             launcher
  workflows/rag-eval/sampling.mjs           strata, allocate, stratifiedSample
  workflows/rag-eval/judge.mjs              judge prompt, parse, judgeAnswer
  workflows/rag-eval/scoring.mjs            scoreRow, aggregate
  workflows/rag-eval/report.mjs             renderSummary, writeReport
  workflows/rag-eval/eval-runner.mjs        runEval
  workflows/rag-eval/rag-eval.js            body
  workflows/rag-eval/main.mjs               CLI launcher
  tests/rag-*.test.mjs
```

---

## Part A — Python retrieval (`rag/`)

All Python commands run from the repo root with the venv active.

### Task 1: Package scaffold, config, chunking

**Files:**
- Create: `rag/__init__.py`, `rag/requirements.txt`, `rag/config.py`, `rag/chunking.py`
- Create: `rag/tests/__init__.py`, `rag/tests/fakes.py`, `rag/tests/conftest.py`, `rag/tests/test_chunking.py`
- Modify: `.gitignore` (add `rag/indexes/`, `__pycache__/`, `.pytest_cache/`)

**Interfaces — produces:**
- `rag.config`: `REPO_ROOT, DATA_DIR, CORPUS_PATH, QRELS_PATH, INDEX_DIR, EMBED_MODEL, RERANK_MODEL, RERANK_MAX_LENGTH=512, RERANK_BATCH_SIZE=32, CHUNK_TOKENS=200, CHUNK_OVERLAP=40, RETRIEVE_DEPTH=50, CONTEXT_K=5, RRF_K=60` (paths derived from `Path(__file__)`).
- `Chunk(chunk_id: str, doc_id: str, text: str)` — frozen dataclass.
- `TokenSpans = Callable[[str], list[tuple[int, int]]]` — character spans of tokens.
- `window_starts(n_tokens, size, overlap) -> list[int]` — raises `ValueError` if `overlap >= size`.
- `chunk_document(doc_id, text, token_spans, size=200, overlap=40) -> list[Chunk]` — chunk text is sliced from the original text by character offsets (keeps case and newlines; the MiniLM tokenizer lowercases).
- `hf_token_spans(tokenizer) -> TokenSpans` — wraps a Hugging Face fast tokenizer with `return_offsets_mapping=True, add_special_tokens=False`.
- Test fakes in `rag/tests/fakes.py`: `whitespace_spans`, `FakeEncoder` (bag-of-words over a fixed 8-word vocab, L2-normalised, accepts `**kwargs`), `FakeReranker` (`predict(pairs, **kwargs)` = count of shared words), `CORPUS` (4 short docs `d1`–`d4`, below). Both fakes **lowercase and split on whitespace**. The hybrid test depends on this: `E370` must match `e370`.

```python
CORPUS = [
    {"doc_id": "d1", "text": "elevator motor overheating fault E307 on tower B"},
    {"doc_id": "d2", "text": "door sensor stuck fault reported by technicians"},
    {"doc_id": "d3", "text": "hvac night schedule setback configuration guide"},
    {"doc_id": "d4", "text": "error E370 indicates elevator door sensor misalignment"},
]
VOCAB = ["elevator", "door", "hvac", "e307", "e370", "motor", "sensor", "schedule"]
```

`rag/requirements.txt`:

```
numpy>=1.26
rank-bm25>=0.2.2
sentence-transformers>=3.0
pytest>=8.0
```

- [ ] **Step 1: Set up the environment**

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: py -m venv .venv; .venv\Scripts\activate
# Ubuntu without ensurepip: sudo apt-get install python3-venv
pip install -r rag/requirements.txt
```

- [ ] **Step 2: Write failing tests** in `rag/tests/test_chunking.py`:
  - `window_starts(500, 200, 40) == [0, 160, 320]`
  - `window_starts(10, 200, 40) == [0]` and `window_starts(200, 200, 40) == [0]`
  - `window_starts(0, 200, 40) == []`
  - `window_starts(10, 5, 5)` raises `ValueError`
  - 500 words `w0 … w499` → chunk IDs `swg1#0..#2`; chunk 0 is `w0..w199`, chunk 1 starts `w160` ends `w359`, chunk 2 starts `w320` ends `w499`
  - `"Title: IBM Fix\n\nRun STREAMTOOL now"` → one chunk whose text equals the input exactly
  - a blank document → `[]`

- [ ] **Step 3: Run — expect FAIL** (`ModuleNotFoundError: rag.chunking`)

```bash
python -m pytest rag/tests/test_chunking.py -q
```

- [ ] **Step 4: Implement** `config.py`, `chunking.py`, `fakes.py`, `conftest.py` (fixtures `corpus`, `fake_encoder`, `fake_reranker`). Core of chunking:

```python
def window_starts(n_tokens, size, overlap):
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if n_tokens == 0:
        return []
    step = size - overlap
    starts = [0]
    while starts[-1] + size < n_tokens:
        starts.append(starts[-1] + step)
    return starts

def chunk_document(doc_id, text, token_spans, size=config.CHUNK_TOKENS, overlap=config.CHUNK_OVERLAP):
    spans = token_spans(text)
    chunks = []
    for n, start in enumerate(window_starts(len(spans), size, overlap)):
        end = min(start + size, len(spans))
        chunks.append(Chunk(f"{doc_id}#{n}", doc_id, text[spans[start][0] : spans[end - 1][1]]))
    return chunks
```

- [ ] **Step 5: Run — expect PASS** (7 tests)
- [ ] **Step 6: Commit** — `feat(rag): add package config and token-window chunking`

---

### Task 2: BM25 tokenizer, model loading, index build and load

**Files:**
- Create: `rag/io_utils.py`, `rag/lexical.py`, `rag/models.py`, `rag/index.py`, `rag/build_index.py`
- Create: `rag/tests/test_lexical.py`, `rag/tests/test_index.py`
- Modify: `rag/tests/conftest.py` (add `index` fixture)

**Interfaces:**
- Consumes: `Chunk`, `chunk_document`, `hf_token_spans`, config constants.
- Produces:
  - `read_jsonl(path) -> Iterator[dict]`, `write_jsonl(path, rows) -> int` (creates parent dirs, UTF-8, `ensure_ascii=False`).
  - `bm25_tokenize(text) -> list[str]` — lowercase, regex `[a-z0-9]+(?:[._-][a-z0-9]+)*`.
  - `load_tokenizer()`, `load_embedder()`, `load_reranker()` — heavy imports inside the functions.
    - `load_tokenizer()` sets `tokenizer.model_max_length` very high. It only finds token boundaries for chunking, and some documents are 27k words, which would otherwise print a "sequence too long" warning on every build.
    - `load_reranker()` = `CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)`. The longest test question is 259 words, so the default length would truncate (question, chunk) pairs.
  - `embed(encoder, texts) -> np.ndarray` — float32, `normalize_embeddings=True`, `batch_size=64`, no progress bar.
  - `StaleIndexError(RuntimeError)`; message always contains `run: python -m rag.build_index`.
  - `Index(chunks: list[Chunk], embeddings: np.ndarray, bm25: BM25Okapi, manifest: dict)`.
  - `index_settings() -> {"embed_model", "chunk_tokens", "chunk_overlap"}` from config.
  - `build_index(docs, token_spans, encoder, out_dir, corpus_sha1, settings=None) -> manifest` — writes `chunks.jsonl`, `embeddings.npy`, `bm25.pkl`, then `manifest.json` **last** (an interrupted build reads as missing). Manifest = settings + `corpus_sha1, n_docs, n_chunks, built_at`. Raises `ValueError` if no chunks.
  - `load_index(index_dir, expected=None) -> Index` — `StaleIndexError` if the manifest is missing, any setting differs from `expected` (default `index_settings()`), or file row counts disagree.
  - `python -m rag.build_index [--corpus PATH] [--out DIR]` — prints the manifest. It also calls `load_reranker()` once, so the reranker is downloaded here, in the user's own shell. Otherwise the first download would happen inside a Fleet `command()` without the user's HF environment.
  - conftest fixture `index`: builds the 4-doc fake index in `tmp_path` and loads it.

- [ ] **Step 1: Write failing tests**
  - `bm25_tokenize("Upgrade to 4.1.1.2 fails: SQLCODE401 (APAR PI19804)") == ["upgrade","to","4.1.1.2","fails","sqlcode401","apar","pi19804"]`
  - `bm25_tokenize("door/sensor, stuck!") == ["door","sensor","stuck"]`
  - build + load round trip on the fake corpus: manifest `n_chunks == 4`, `n_docs == 4`, `corpus_sha1 == "abc"`; chunk IDs `d1#0..d4#0`; embeddings shape `(4, 8)`, dtype float32, row norms ≈ 1; `bm25.get_scores(["e307"])` has 4 values
  - `load_index(tmp_path / "missing")` raises `StaleIndexError` matching `build_index`
  - `load_index(dir, expected={...index_settings(), "chunk_tokens": 999})` raises `StaleIndexError` matching `chunk_tokens`
  - a corpus of one blank doc → `ValueError`
- [ ] **Step 2: Run — expect FAIL**: `python -m pytest rag/tests -q`
- [ ] **Step 3: Implement** the five modules and the fixture.
- [ ] **Step 4: Run — expect PASS** (all tests so far)
- [ ] **Step 5: Commit** — `feat(rag): build and load the chunk index (MiniLM vectors + BM25)`

---

### Task 3: Shared retrieval helpers and the naive retriever

**Files:**
- Create: `rag/retrievers/__init__.py`, `rag/retrievers/base.py`, `rag/retrievers/naive.py`
- Create: `rag/tests/test_naive.py`

**Interfaces:**
- Consumes: `Index`, `embed`, `bm25_tokenize`, `Chunk`.
- Produces:
  - `ScoredChunk(chunk: Chunk, score: float)` — frozen dataclass.
  - `top_indices(scores, depth) -> list[int]` — `np.argsort(-scores, kind="stable")[:depth]`.
  - `dense_search(index, encoder, query, depth) -> list[ScoredChunk]` — cosine via `index.embeddings @ query_vector`.
  - `bm25_search(index, query, depth) -> list[ScoredChunk]` — drops chunks with score ≤ 0.
  - `doc_ranking(doc_ids) -> list[str]` — first occurrence of each doc, in order.
  - `to_result(method, query, scored) -> {"method","query","chunks":[{"rank","chunk_id","doc_id","score","text"}]}` — rank from 1, score rounded to 6 places.
  - `NaiveRetriever(index, encoder)`: `name = "naive"`, `retrieve(query, depth=50) -> list[ScoredChunk]` = `dense_search`.
  - `METHODS = ("naive",)`; `make_retriever(method, index, encoder, reranker=None)` — `ValueError("unknown method …")` otherwise.

- [ ] **Step 1: Write failing tests**
  - `NaiveRetriever(index, fake_encoder).retrieve("hvac schedule", depth=2)` → 2 hits, first is `d3`, scores descending
  - `retrieve("door", depth=50)` → 4 hits (capped by index size)
  - `doc_ranking(["a","b","a","c","b"]) == ["a","b","c"]`
  - `to_result` on 3 hits → ranks `[1,2,3]`, chunk keys exactly `{rank, chunk_id, doc_id, score, text}`
  - `make_retriever("naive", …).name == "naive"`; `make_retriever("graph", …)` raises `ValueError` matching `unknown method`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(rag): add naive dense retriever`

---

### Task 4: Hybrid retriever (dense + BM25 → RRF → cross-encoder)

**Files:**
- Create: `rag/retrievers/hybrid.py`, `rag/tests/test_hybrid.py`
- Modify: `rag/retrievers/__init__.py` — `METHODS = ("naive", "hybrid")`; `make_retriever("hybrid", …)` builds `HybridRetriever`, and raises `ValueError("the hybrid method needs a reranker")` when `reranker is None`.

**Interfaces — produces:**
- `rrf_fuse(rankings: list[list[str]], k=60) -> list[tuple[str, float]]` — `score = Σ 1/(k + rank)`, rank from 1; sorted by score desc, ties by first appearance.
- `HybridRetriever(index, encoder, reranker, pool=50)`: `name = "hybrid"`; `retrieve(query, depth=50)`:
  1. dense top `pool`, BM25 top `pool`
  2. `rrf_fuse` of the two chunk-ID lists, keep top `pool`
  3. `reranker.predict([(query, chunk.text), …], batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)`
  4. return the top `depth` by reranker score as `ScoredChunk`s (score = reranker logit)

```python
def rrf_fuse(rankings, k=config.RRF_K):
    scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk_id, len(first_seen))
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
```

- [ ] **Step 1: Write failing tests**
  - `rrf_fuse([["a","b","c"],["c","a"]], k=60)` → order `a, c, b`; scores `1/61+1/62`, `1/63+1/61`, `1/62`
  - `rrf_fuse([["x"],["y"]])` → `x, y` (tie keeps first seen)
  - `bm25_search(index, "E307", 10)` → only `d1` (exact code match; zero-score chunks dropped)
  - `HybridRetriever(...).retrieve("E370 elevator door sensor", depth=3)` → 3 hits, first `d4`, scores descending
  - `make_retriever("hybrid", …, fake_reranker).name == "hybrid"`; without a reranker → `ValueError` matching `reranker`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(rag): add hybrid retriever with RRF fusion and cross-encoder rerank`

---

### Task 5: `retrieve` command-line tool

**Files:**
- Create: `rag/retrieve.py`, `rag/tests/test_retrieve_cli.py`

**Interfaces — produces:**

```
python -m rag.retrieve --method naive|hybrid (--query TEXT | --query-file PATH | --queries-file JSONL --out JSONL)
                       [--depth 50] [--index-dir rag/indexes]
```

- `--query-file` (UTF-8 text file) exists so the Node side never has to shell-quote a question with newlines and quotes.
- Single mode prints one `to_result` object to stdout with `json.dumps` default (`ensure_ascii=True`, so any console encoding works).
- Batch mode reads `{query_id, text}` rows, writes one row per query = `{"query_id", **to_result(...)}`, prints `{"method","queries","out","depth","embed_model","rerank_model"}` (`rerank_model` is `null` for naive). The eval stores this in `run.json`, so a report records which models produced it.
- `--queries-file` without `--out` → argparse error (exit 2).
- `StaleIndexError` → message on stderr, exit code 2. `load_index` runs before any model loads, so this check is fast.
- `main(argv=None, retriever_loader=load_retriever) -> int` — tests inject a loader that builds retrievers from the fake index.

- [ ] **Step 1: Write failing tests** (with a `loader` fixture that calls `make_retriever` on the fake index)
  - `--method naive --query "hvac schedule" --depth 2` → exit 0, 2 chunks, first `d3`
  - `--method hybrid --query-file q.txt` containing `E370 elevator door sensor` → first chunk `d4`
  - batch over two queries with `--depth 3` → output rows `q1, q2`, each `method == "naive"` with 3 chunks; stdout `queries == 2` and `rerank_model` is `null`
  - `--queries-file` without `--out` → `SystemExit`
  - default loader with `--index-dir <missing>` → exit 2, stderr contains `build_index`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(rag): add retrieve CLI with single, file and batch modes`

---

### Task 6: Retrieval metrics

**Files:**
- Create: `rag/retrieval_metrics.py`, `rag/tests/test_retrieval_metrics.py`

**Interfaces — produces:**
- `load_qrels(path) -> dict[str, set[str]]`
- `gold_rank(ranked_doc_ids, gold: set[str]) -> int | None` — 1-based.
- `compute_metrics(results, qrels, ks=(1, 3, 5, 10)) -> {"method", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr"}` — only rows whose `query_id` has a qrel; document rank from `doc_ranking` over the chunk list; a missing gold doc counts 0; values rounded to 4 places; `n == 0` gives zeros.
- `python -m rag.retrieval_metrics --results JSONL [--qrels PATH]` — prints that JSON.

- [ ] **Step 1: Write failing tests** with four result rows:
  - `q1` chunks `[g1, x]` (gold `g1`) → rank 1
  - `q2` chunks `[x, x, y, g2]` (gold `g2`) → document rank 3 (the duplicate `x` must not count twice)
  - `q3` chunks `[x, y]` (gold `g3`) → not found
  - `q4` has no qrel (unanswerable) → ignored
  - Expected: `n == 3`, `hit@1 == 0.3333`, `hit@3 == hit@5 == hit@10 == 0.6667`, `mrr == 0.4444`
  - CLI run on files in `tmp_path` prints the same numbers
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(rag): add Hit@k and MRR retrieval metrics`

---

## Part B — Agent kit (`agent/`)

All Node commands run inside `agent/`.

### Task 7: Scaffold the kit project and shared helpers

**Files:**
- Create (generated): `agent/` via the kit generator
- Create: `agent/workflows/rag-shared/paths.mjs`, `agent/workflows/rag-shared/llm-json.mjs`
- Create: `agent/tests/rag-paths.test.mjs`, `agent/tests/rag-llm-json.test.mjs`

**Interfaces — produces:**
- `paths.mjs`:
  - `REPO_ROOT` = three levels up from `agent/workflows/rag-shared/`
  - `TEST_QUERIES_PATH`, `QRELS_PATH` (under `RAG_Dataset/data/processed/`), `INDEX_MANIFEST_PATH` (`rag/indexes/manifest.json`), `REPORTS_DIR` = `agent/evals/reports/rag`
  - `pythonBin(platform = process.platform, root = REPO_ROOT)` → `<root>/.venv/bin/python`, or `<root>\.venv\Scripts\python.exe` on `win32`. It uses `path.posix` / `path.win32` explicitly, so the tests give the same result on any OS.
  - `quotePath(p)` → `'p'`; throws if `p` contains `'` or a newline. Single quotes are literal in both `sh` and PowerShell, so `$` and spaces are safe.
  - `pythonModuleCommand(moduleName, args, { platform, root })`:
    - POSIX: `cd '<root>' && '<python>' -m <module> <args…>`
    - `win32`: `Set-Location -LiteralPath '<root>'; & '<python>' -m <module> <args…>` (PowerShell, matching what Fleet runs on Windows)
- `llm-json.mjs`:
  - `toText(reply)` — accepts a string, `{ text }`, `{ content: [{ text }] }`, or `{ output }`
  - `extractJsonObject(text)` — strips code fences, scans for brace-balanced objects (respecting JSON strings), returns the **last** one that parses, or `null`

- [ ] **Step 1: Scaffold**

```bash
npm create @apralabs/agent-kit@0.3.0 agent -- --no-install
cd agent && npm install && npm test        # the kit's own tests must pass first
```

The generated `agent/.gitignore` already ignores `evals/reports/` and `node_modules/`.

- [ ] **Step 2: Write failing tests**
  - `REPO_ROOT` contains a `RAG_Dataset` directory
  - `pythonBin('linux', '/r') === '/r/.venv/bin/python'`; `pythonBin('win32', 'C:\\r')` ends with `.venv\Scripts\python.exe`
  - `pythonModuleCommand('rag.retrieve', ['--method','naive'], { platform: 'linux', root: '/r' }) === "cd '/r' && '/r/.venv/bin/python' -m rag.retrieve --method naive"`
  - `pythonModuleCommand('rag.retrieve', [], { platform: 'win32', root: 'C:\\r' }) === "Set-Location -LiteralPath 'C:\\r'; & 'C:\\r\\.venv\\Scripts\\python.exe' -m rag.retrieve"`
  - `quotePath("a'b")` and a path with a newline throw; `quotePath('/a b $HOME') === "'/a b $HOME'"`
  - `extractJsonObject`: plain JSON; a ```json fenced block; prose before and after; braces inside strings (`{"answer":"use {x}"}`); two objects returns the last; no object returns `null`
  - `toText` for each accepted shape
- [ ] **Step 3: Run — expect FAIL**: `node --test tests/rag-paths.test.mjs tests/rag-llm-json.test.mjs`
- [ ] **Step 4: Implement**
- [ ] **Step 5: Run — expect PASS**; also `npm test` (kit tests still green)
- [ ] **Step 6: Commit** — `feat(agent): scaffold Apra agent kit project with RAG path and JSON helpers`

---

### Task 8: Retrieval client, answer generation, `rag-answer` workflow and tool

**Files:**
- Create: `agent/workflows/rag-shared/retrieval-client.mjs`, `agent/workflows/rag-shared/generation.mjs`
- Create: `agent/workflows/rag-answer/rag-answer.js`, `agent/workflows/rag-answer/main.mjs`
- Create: `agent/tests/rag-generation.test.mjs`, `agent/tests/rag-answer.test.mjs`
- Create: `agent/tests/rag-fakes.mjs` — shared test helpers `fakeAgent(replies | fn)` and `fakeCommand(result | fn)` (both record their calls), and `chunk(docId, n)`
- Modify: `agent/mcp/registry.mjs` (add `rag_answer`), `agent/package.json` (`"rag:answer": "node workflows/rag-answer/main.mjs"`)
- Modify: `agent/host.config.mjs` — replace the template `description` / `agentDescription` placeholders (spec §3). The text must say that the agent answers IBM support questions from the TechQA technotes and must **always** use `rag_answer` rather than answering from memory. It passes the full question including error messages, uses `hybrid` unless the user asks for `naive`, and when the tool abstains it tells the user the technotes don't cover the question.

**Interfaces:**
- Consumes: `pythonModuleCommand`, `quotePath`, `extractJsonObject`, `toText`.
- Produces — `retrieval-client.mjs`:
  - `RAG_METHODS = ['naive', 'hybrid']`, `RETRIEVE_DEPTH = 50`
  - `readJsonl(file)`, `writeJsonl(file, rows)`
  - `createRetrievalClient(command)` → `{ single, batch, metrics }`. Every call uses `{ member_name: 'doer', failSoft: true, timeout_s: 3600, timeoutMs: 3_700_000 }` and throws `Error('<label> failed: <error>')` when `ok` is false.
    - `single({ method, question, depth })` → chunk array. Writes the question to a temp file and passes `--query-file`; removes the temp dir afterwards.
    - `batch({ method, rows, outFile, depth })` → `{ summary, chunksById: Map<query_id, chunks> }`, where `summary` is the JSON line the CLI prints. It writes `rows` to `<outFile minus .jsonl>.queries.jsonl` and passes `--queries-file` / `--out`.
    - `metrics({ resultsFile, qrelsFile })` → parsed metrics object.
- Produces — `generation.mjs`:
  - `CONTEXT_K = 5`; `GENERATOR = { member_name: 'doer', model: 'standard' }`
  - `formatContext(chunks)` → `[i] doc_id=<id>\n<text>` blocks joined by `\n\n---\n\n` (also used by the judge)
  - `buildAnswerPrompt(question, chunks)`: answer only from the context; if not covered, set `abstained: true` with a one-line refusal; reply with only `{"answer": "...", "abstained": true|false, "cited_doc_ids": [...]}`
  - `parseAnswer(text)` → `{ ok: true, value: { answer, abstained, cited_doc_ids } }` or `{ ok: false, error }`; checks the types of all three fields
  - `askForJson({ agent, prompt, options, parse, failure })` — the shared "ask, parse, retry once with the parse error appended" loop. It returns `{ ...value, calls }` or `{ error: failure, raw, calls }`. The judge (Task 10) reuses it.
  - `generateAnswer({ agent, question, chunks })` → `askForJson` with `GENERATOR` and failure `generation_parse_failed`. Fleet errors propagate (the caller decides).
- Produces — `rag-answer.js` body: `args = { method = 'hybrid', question }` →
  - `{ error: 'unknown_method' }` or `{ error: 'missing_question' }` on bad input
  - `{ method, question, error: 'retrieval_failed', detail }` when retrieval fails (no LLM call)
  - otherwise `{ method, question, context_doc_ids, context_chunk_ids, ...generateAnswer result }`
- `main.mjs`: same pattern as `workflows/hello/main.mjs`; exports `runRagAnswer({ fleetApi, workspace, signal, reportPhase, method, question })`; CLI `node workflows/rag-answer/main.mjs <method> "<question>"`.
  - It calls `engine.executeFile(path.join(here, 'rag-answer.js'), …)` with an absolute path.
  - It loads `standalone.mjs` with a dynamic `import()` inside the CLI branch. The registry and tests import `runRagAnswer`, and must not start Fleet when they do.
- Registry entry `rag_answer`:
  - `inputSchema`: `question: z.string()`, `method: z.enum(['naive','hybrid']).optional()`
  - `annotations: { readOnlyHint: true }`
  - `routing`: `{ description: 'Answer an IBM product support question from the technote collection.', args: { question: { extract: '…' }, method: { extract: '…' } } }`
  - `run` calls `runRagAnswer`
  - the description says it spends LLM tokens and explains the two methods

- [ ] **Step 1: Write failing tests** (fake `command` returns `{ ok, output, error }`; fake `agent` pops queued replies and records `{ prompt, member_name, model }`)
  - `parseAnswer`: valid → ok; missing `answer`, non-boolean `abstained`, non-array `cited_doc_ids` → each `ok: false`
  - `buildAnswerPrompt` contains the question and every `doc_id`
  - `rag-answer` happy path, with retrieval returning 6 chunks `d1..d6`:
    - the command uses `member_name: 'doer'` and contains `-m rag.retrieve` and `--method hybrid`
    - the prompt contains `doc_id=d5` but not `doc_id=d6`
    - the agent is called with `member_name: 'doer'` and `model: 'standard'`
    - the result has the answer and 5 `context_doc_ids`
  - first reply `not json`, second valid → result ok, `calls === 2`
  - two invalid replies → `error === 'generation_parse_failed'`
  - command returns `{ ok: false, error: 'exit 2: no index' }` → `error === 'retrieval_failed'` and no agent call
  - `method: 'graph'` → `unknown_method`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement** the modules, the launcher, the registry entry and the npm script
- [ ] **Step 4: Run — expect PASS**: `npm test`
- [ ] **Step 5: Commit** — `feat(agent): add rag-answer workflow and rag_answer tool`

---

### Task 9: Stratified sampling

**Files:**
- Create: `agent/workflows/rag-eval/sampling.mjs`, `agent/tests/rag-sampling.test.mjs`

**Interfaces — produces:**
- `STRATA = ['answerable_ok', 'answerable_flagged', 'unanswerable']`
- `stratumOf(q)` — `is_impossible` → `unanswerable`; `quality_ok === false` → `answerable_flagged`; otherwise `answerable_ok`
- `mulberry32(seed) -> () => number` — deterministic PRNG in [0, 1)
- `randomSeed()` — `crypto.randomInt(1, 2**31 - 1)`
- `allocate(sizes, total)` — largest-remainder rounding. Ties go in `STRATA` order. No stratum exceeds its size. Throws if `total < 1`, `total` is not an integer, or `total` exceeds the population.
- `stratifiedSample(questions, size, seed) -> { seed, counts, questions }` — each stratum is sorted by `query_id`, Fisher–Yates shuffled with a single `mulberry32(seed)` (strata in `STRATA` order), then the first `counts[k]` are taken.

- [ ] **Step 1: Write failing tests**
  - `allocate({ answerable_ok: 173, answerable_flagged: 10, unanswerable: 90 }, 40)` → `{25, 2, 13}`
  - the same sizes with 10 → `{6, 1, 3}`; with 273 → `{173, 10, 90}`
  - totals `0`, `274` and `2.5` throw
  - a synthetic 273-question set (173 / 10 / 90), sample size **40**: seed 7 twice gives identical `query_id`s; seeds 7 and 8 differ (at size 273 every seed picks the same set); per-stratum counts match `allocate`; no duplicates
  - `stratumOf` covers all three cases
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(eval): add seeded stratified sampling`

---

### Task 10: Judge and scoring

**Files:**
- Create: `agent/workflows/rag-eval/judge.mjs`, `agent/workflows/rag-eval/scoring.mjs`
- Create: `agent/tests/rag-judge.test.mjs`, `agent/tests/rag-scoring.test.mjs`

**Interfaces:**
- Consumes: `formatContext`, `extractJsonObject`, `toText`, `STRATA`, `stratumOf`.
- Produces — `judge.mjs`:
  - `JUDGE = { member_name: 'reviewer', model: 'premium' }`; `CORRECTNESS_LEVELS = ['full', 'partial', 'none']`
  - `buildJudgePrompt({ question, chunks, answer, goldAnswer })`. It instructs the judge to:
    1. split the answer into factual claims and mark each supported or not using **only the retrieved context**
    2. rate whether the answer **contains the information in the reference** (full / partial / none), judging containment rather than wording, with extra correct detail allowed
    3. reply with only `{"claims":[{"text","supported"}], "correctness", "reason"}`
  - `parseJudge(text)` → `{ ok, value: { claims, correctness, reason } }` or `{ ok: false, error }`; `reason` defaults to `''`
  - `judgeAnswer({ agent, question, chunks, answer, goldAnswer })` → `askForJson` with `JUDGE` and failure `judge_error`: value + `calls`, or `{ error: 'judge_error', raw, calls }`
- Produces — `scoring.mjs`:
  - `CORRECTNESS_SCORE = { full: 1, partial: 0.5, none: 0 }`
  - `scoreRow({ question, goldDocIds, contextDocIds, generation, judgment })` → `{ stratum, errors[], abstained, context_hit, correctness, faithfulness, zero_claims, wrong_refusal, abstention_correct }`.

    **Every field starts as `null`** (`errors: []`, `zero_claims: false`), and only the rules below set it. `null` means "not measured for this row" and is skipped by the means. A field left `undefined`, or set only to `true`, breaks the rates. For example, if answered rows never set `wrong_refusal: false`, the wrong-refusal rate is always 1.0.

    Rules, from spec §6.3:
    - **Answerable:** `context_hit` = any gold doc among the context docs (`null` for unanswerable).
    - **Generation error:** push the error name; stop. Everything else stays `null`.
    - **Unanswerable:** `abstention_correct = abstained`; stop.
    - **Answerable:** `wrong_refusal = abstained`, so `false` when it answered.
    - **Answerable and abstained:** `correctness = 0`, no judge.
    - **Answerable and answered:**
      - judge error → push it and stop (`correctness` stays `null`)
      - otherwise `correctness = CORRECTNESS_SCORE[...]`
      - with 0 claims: `zero_claims = true` and `faithfulness = null`
      - otherwise `faithfulness = supported / total`, unrounded
  - `aggregate(rows: [{ method, scores }])` → `{ [method]: { [stratum]: summary } }`. Each summary holds:
    - `n`, `errors` (counts by name)
    - `correctness`, `faithfulness`, `faithfulness_n`, `zero_claims` (a count)
    - `wrong_refusal_rate`, `context_hit_rate`
    - `abstention_accuracy`, and `hallucination_rate = 1 − abstention_accuracy`, which is `null` when `abstention_accuracy` is `null`. In JavaScript `1 - null === 1`, so this needs an explicit check.
    
    Means skip `null`; an empty list gives `null`.

- [ ] **Step 1: Write failing tests**
  - `parseJudge`: valid; bad `correctness` value; `supported` not a boolean; no JSON → not ok
  - `judgeAnswer`: called with `reviewer` / `premium`; a garbage reply then a valid one → ok with `calls === 2`; two garbage replies → `judge_error`
  - `scoreRow` cases:
    - answered, `full`, 3 claims with 2 supported → correctness 1, faithfulness ≈ 2/3 (compare with a tolerance), wrong_refusal **false**
    - abstained answerable → correctness 0, wrong_refusal true, faithfulness null
    - judge error → errors `['judge_error']`, correctness null
    - zero claims → `zero_claims` true, faithfulness null
    - unanswerable abstained → `abstention_correct` true
    - unanswerable answered → false
    - generation error → errors `['generation_parse_failed']`
    - context hit true and false
  - `aggregate` over four rows for one method: answered `full` with 1 of 2 claims supported, abstained answerable, judge error, and an unanswerable question that was answered. Expected:
    - `answerable_ok`: `n` 3, correctness 0.5, faithfulness 0.5 (`faithfulness_n` 1), **`wrong_refusal_rate` 1/3**, errors `{ judge_error: 1 }`, **`hallucination_rate` `null`**
    - `unanswerable`: abstention accuracy 0, hallucination rate 1
    - `answerable_flagged`: `n` 0, correctness `null`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(eval): add LLM judge and Ragas-style scoring`

---

### Task 11: Report writer

**Files:**
- Create: `agent/workflows/rag-eval/report.mjs`, `agent/tests/rag-report.test.mjs`

**Interfaces — produces:**
- `renderSummary({ run, retrieval, answers, calls }) -> string` (Markdown):
  - **Title:** `# RAG eval — <run.started_at>`, then the methods and git commit.
  - **`## Retrieval (all answerable test questions, no LLM)`:** a table with `Method | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR`.
  - **If `answers` is null:** the line `_Retrieval-only run: answer quality not measured._`
  - **Otherwise, `## Answer quality (sample of N, seed S: a / b / c)`** with one section per stratum, in `STRATA` order:
    - Answerable strata: `Method | n | Correctness | Faithfulness (n) | Zero-claim answers | Wrong refusals | Context hit | Errors`
    - Unanswerable: `Method | n | Abstention accuracy | Hallucination rate | Errors`
    - The flagged section's heading says it is not a headline number.
  - **`## Calls`:** `Method | Generation calls | Judge calls | Total time (s) | Median generation (s)`
  - **`## Notes`:** the generator and judge members and tiers, the same-provider caveat, and the `--seed` tip.
  - **Formatting:** numbers to 3 decimal places, `null` as `–`, and errors as `name: count` joined by commas, or `0`.
- `writeReport(dir, { run, retrieval, answers, calls, rows })` — writes `run.json` (pretty), `results.jsonl`, and `summary.md`; returns their paths. `retrieval/<method>.jsonl` is written by Python, not here.

- [ ] **Step 1: Write failing tests**
  - a retrieval-only render contains the retrieval table rows and the retrieval-only line, and no "Answer quality"
  - a full render contains each method row, the seed, the three stratum headings, the "not a headline" wording, and `–` for a null value
  - `writeReport` into `tmp` creates the three files; `results.jsonl` has one line per row; `run.json` round-trips
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit** — `feat(eval): add Markdown and JSON report writer`

---

### Task 12: Eval runner, `rag-eval` workflow and CLI

**Files:**
- Create: `agent/workflows/rag-eval/eval-runner.mjs`, `agent/workflows/rag-eval/rag-eval.js`, `agent/workflows/rag-eval/main.mjs`
- Create: `agent/tests/rag-eval.test.mjs`
- Modify: `agent/package.json` (`"rag:eval": "node workflows/rag-eval/main.mjs"`)

**Interfaces:**
- Consumes: everything from Tasks 7–11.
- Produces — `eval-runner.mjs`:
  - `loadQrels(file) -> Map<query_id, doc_id[]>`
  - `runEval({ agent, log, retrieval, options, now = Date.now })`. `options` = `{ methods, sampleSize, seed, retrievalOnly, reportDir, testQueriesPath, qrelsPath, gitCommit, indexManifest }`. It:
    1. validates the methods, then reads the test questions and qrels
    2. for each method, runs **one** `retrieval.batch` over **all 273** test questions into `<reportDir>/retrieval/<method>.jsonl`, then `retrieval.metrics` on that file. `compute_metrics` already scores only questions with a qrel, so the metrics still cover the 183 answerable ones. The answer sample reuses these results, which means one model load per method and no separate sample retrieval files.
    3. if `retrievalOnly`: writes the report with `answers: null` and returns `{ reportDir, retrieval }`
    4. otherwise uses `seed ?? randomSeed()` and calls `stratifiedSample`
    5. for each method, for each sampled question:
       - takes the top `CONTEXT_K` chunks from step 2's results and calls `generateAnswer`
       - calls `judgeAnswer` only if the question is answerable, generation succeeded and the answer did not abstain
       - wraps each call so a thrown Fleet error becomes `{ error: 'fleet_error', detail, calls: 1 }`. The `calls: 1` keeps the call counts numbers rather than `NaN`. Errors with `code` `CANCELLED` or `BUDGET_EXCEEDED` are **re-thrown**, so a stop request ends the run instead of turning every remaining row into an error.
       - records timings, then `scoreRow`
    6. aggregates, writes the report, and returns `{ reportDir, retrieval, answers, calls }`

    `run.json` holds (spec §6.4):
    - `started_at`, `finished_at`, `methods`, `retrieval_only`, `git_commit`
    - `sample: { size, seed, counts, query_ids }`
    - `settings: { retrieve_depth, context_k, generator, judge, index, retrieval }`:
      - `index` is the index manifest: embedding model, chunk size and overlap, corpus hash, chunk count.
      - `retrieval[method]` is that method's batch `summary`: embedding and reranker model, depth.

    Each row in `results.jsonl` holds:
    - `query_id`, `method`, `stratum`
    - `context_chunk_ids`, `context_doc_ids`, `gold_doc_ids`
    - `generation`, `generation_ms`, `judgment`, `judge_ms`, `scores`

    `calls[method]` holds `{ generation_calls, judge_calls, total_ms, median_generation_ms }`.
- `rag-eval.js` body: `main(context)` → `runEval({ agent, log, retrieval: createRetrievalClient(command), options: args.options })`.
- `main.mjs`:
  - `parseCli(argv)` → `{ methods, sampleSize, seed, retrievalOnly, reportDir }`. Defaults: methods `naive,hybrid`, sample 40, seed `null`. Throws on a non-integer or non-positive sample, or a non-integer seed.
  - `localCommand(cmd)` → `spawnSync`, returning `{ ok, output, error }`. It runs the command through the system shell on POSIX and through `powershell.exe -NoProfile -NonInteractive -Command` on Windows, matching the command form `pythonModuleCommand` builds.
  - `runRagEval({ fleetApi, workspace, signal, reportPhase, options })`:
    - **`retrievalOnly`:** calls the body directly with `localCommand` and an `agent` that throws. No Fleet, no token.
    - **Otherwise:** fails fast if `CLAUDE_CODE_OAUTH_TOKEN` is unset, with the message `set CLAUDE_CODE_OAUTH_TOKEN or use --retrieval-only`. Then it uses `withStandaloneLease` (imported dynamically so the tests can import `parseCli`) and `engine.executeFile(path.join(here, 'rag-eval.js'), { …, options })`.
  - **CLI entry:** the default `reportDir` is `REPORTS_DIR/<ISO timestamp with : and . replaced by ->`. `gitCommit` comes from `git rev-parse HEAD` at the repo root, with stderr ignored, or `null`. `indexManifest` is read from `INDEX_MANIFEST_PATH`, or `null`. It prints the path to `summary.md`.

- [ ] **Step 1: Write failing tests**
  - `parseCli([])` gives the defaults; `parseCli(['--methods','naive','--sample','10','--seed','7','--retrieval-only'])` parses all four; `--sample 0` and `--seed x` throw
  - **End to end** with a fake retrieval client and fake agent:
    - Data, in `tmp`: test questions `q1` (answerable_ok, gold `g1`), `q2` (answerable_ok, gold `g2`), `q3` (unanswerable); `qrels` `q1→g1`, `q2→g2`.
    - Fake `batch` records which IDs it was asked for, writes `outFile`, and returns `{ summary, chunksById }`. In `chunksById`, `q1`'s context contains `g1` and `q2`'s does not.
    - Fake `metrics` returns a fixed object.
    - Fake `doer`: `q1` and `q2` answered, `q3` abstained.
    - Fake `reviewer`: for `q1`, `full` with 2 claims, 1 of them supported; for `q2`, garbage twice.
    - Options: methods `['naive','hybrid']`, sample 3, seed 7.
    - Retrieval: exactly two batches, one per method, each over `q1, q2, q3`.
    - Assertions for each method:
      - `answerable_ok`: `n === 2`, `correctness === 1`, `faithfulness === 0.5`, `errors.judge_error === 1`, `context_hit_rate === 0.5`
      - `unanswerable`: `abstention_accuracy === 1`
    - Calls: 6 `reviewer` calls in total. `calls.naive.judge_calls === 3`: one for `q1`, and two for `q2` because of the retry.
    - Files: `summary.md`, `results.jsonl` (6 rows), `run.json` and `retrieval/naive.jsonl` all exist. `run.json` has `sample.seed === 7`, 3 `query_ids`, `settings.index` and `settings.retrieval.hybrid.rerank_model`.
  - **Retrieval-only:** an agent that throws is never called; the result has no `answers`; `summary.md` contains the retrieval-only line.
  - **Fleet error:** an agent that throws for `q1` gives that row the error `fleet_error`, and the run still completes.
  - **Stop request:** an agent that throws an error with `code: 'CANCELLED'` makes `runEval` reject.
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run — expect PASS**: `npm test`
- [ ] **Step 5: Commit** — `feat(eval): add rag-eval workflow and CLI`

---

### Task 13: README and manual live smoke test

**Files:**
- Modify: `README.md` (repo root)

**README sections:**
1. **What this is:** two RAG strategies plus an eval, with links to the spec and this plan.
2. **Setup:**
   - Python: venv, then `pip install -r rag/requirements.txt`.
   - Optional, for a CPU-only machine: install the CPU build of torch first with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
   - `python -m rag.build_index`. Run it in your own shell: it also downloads the reranker. Fleet members do not see `HF_HOME`, `HF_TOKEN` or proxy variables, so the models must already be in the default Hugging Face cache.
   - Node: `cd agent && npm install`
   - Fleet and Claude CLIs, needed for anything except `--retrieval-only`. `--no-install` skipped these during scaffolding:

     ```bash
     npm i -g @apralabs/apra-fleet @anthropic-ai/claude-code
     apra-fleet install --skill none
     export CLAUDE_CODE_OAUTH_TOKEN="$(claude setup-token)"
     npm run doctor            # the kit's environment check
     ```
3. **Use:**
   - `python -m rag.retrieve --method hybrid --query "..."`
   - `npm run rag:answer -- hybrid "..."`
   - `npm run rag:eval -- --retrieval-only`
   - `npm run rag:eval -- --sample 40 [--seed N]`
   - where reports go (`agent/evals/reports/rag/<timestamp>/`)
4. **Tests:** `python -m pytest rag/tests -q` and `cd agent && npm test`.
5. **Reading a report:** the three strata, why flagged is not a headline, and the judge caveat.

- [ ] **Step 1: Write the README**
- [ ] **Step 2: Live smoke, by hand, not in CI**
  1. `python -m rag.build_index` → the manifest shows `n_docs: 496` and `n_chunks: 5015`, and no "sequence too long" warning.
  2. `python -m rag.retrieve --method hybrid --query "How do I set an environment variable for a Streams application?" --depth 5` → `swg21996508` is rank 1 (it is the only qrel of test question `TRAIN_Q000`).
  3. `cd agent && npm run rag:eval -- --retrieval-only` → `summary.md` has both methods with `n = 183` and the numbers in "Verified" below.
  4. `npm run rag:eval -- --sample 4 --seed 1` (needs the token) → the report has 4 questions × 2 methods, and the error counts are 0 or explained.
  5. Record the retrieval numbers and the smoke results in the PR description.
- [ ] **Step 3: Commit** — `docs: add setup, usage and eval guide`

---

## Self-review

- **Spec coverage:**

  | Spec section | Covered by |
  | --- | --- |
  | §4.1 Chunking | Task 1 |
  | §4.2 Index build | Task 2 |
  | §4.3 Strategies | Tasks 3–4 |
  | §4.4 Result shape | Task 3 |
  | §4.5 Retrieve CLI | Task 5 |
  | §4.6 Retrieval metrics | Task 6 |
  | §5 Answering | Tasks 7–8 |
  | §6.1 and §6.2 Command and retrieval metrics | Task 12 |
  | §6.3 Sampling | Task 9 |
  | §6.3 Judge and metrics | Task 10 |
  | §6.4 Output | Task 11 |
  | §6.5 Expected cost | Reported in Task 12 (`calls`) |
  | §3 Agent setup (`host.config.mjs`) | Task 8 |
  | §7 Errors and limits | Tasks 2, 5, 8, 10 and 12 |
  | §8 Testing | Every task, plus Task 13's smoke test |
  | §9 Failure analysis | Deliberately excluded |

- **Additions beyond the spec:**
  - `--query-file` on the retrieve CLI (spec §4.5 lists `--query` only). It's needed to pass multi-line questions safely from Node.
  - `formatContext`, shared by the answer prompt and the judge prompt; `askForJson`, shared by generation and the judge.
  - Retrieval runs once per method over all 273 test questions, rather than once over the 183 answerable ones plus again over the sample (spec §6.2's wording). The metrics are identical, and it saves a model load and two files per method.
  - Spec §8 says the Node tests use the kit's scripted Fleet. They use fake `command` / `agent` functions instead (`tests/rag-fakes.mjs`), the same approach as the kit's own `tests/hello.test.mjs`. That way no Fleet process starts in tests.
- **Names checked across tasks:**
  - `createRetrievalClient().batch` returns `{ summary, chunksById }`, which `runEval` consumes
  - `generateAnswer` and `judgeAnswer` both return `calls`
  - `scoreRow` and `aggregate` field names match what `renderSummary` reads
  - `STRATA` is defined once, in `sampling.mjs`

## Verified

Every task above was implemented once in a scratch copy, outside this repo, to check the plan before review. The checks:

- **Unit tests:** all pass. Python has 34 tests. Node has 55: the kit's own plus 29 RAG tests.
- **Real models, CPU only:**
  - `build_index`: 496 documents → 5,015 chunks, about 1 minute.
  - The smoke question in Task 13 step 2 returns `swg21996508` at rank 1 for both methods.
- **`npm run rag:eval -- --retrieval-only` on the real index:** about 4 minutes for both methods over all 273 questions. The run was through the agent launcher and the Python CLIs, without Fleet.

  | Method | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
  | --- | --- | --- | --- | --- | --- | --- |
  | naive | 183 | 0.792 | 0.891 | 0.923 | 0.945 | 0.849 |
  | hybrid | 183 | 0.792 | 0.885 | 0.918 | 0.973 | 0.850 |

  Hybrid + rerank does not beat naive at the top ranks on this corpus. It only finds more gold documents by rank 10. With `CONTEXT_K = 5`, the answer-quality eval will show whether that matters.
- **Not run:** the Fleet steps (`rag:answer`, and `rag:eval` with a sample). They need the Fleet CLI and a Claude token.

**Independent review.** A separate review of the plan against the kit and Fleet source led to these changes:

- **Explicit `scoreRow` defaults.** Without them the wrong-refusal rate always reads 1.0, and the hallucination rate reads 1 for answerable questions.
- **Windows commands in PowerShell form.** Fleet runs Windows member commands in PowerShell, not `cmd`.
- **Cancellation and budget errors are re-thrown.**
- **`calls: 1` on Fleet errors.**
- **Absolute `executeFile` paths.**
- **Model names and the index manifest in `run.json`.**
- **TechQA `host.config.mjs` text.**
- **Global Fleet and Claude CLI setup, and HF environment caveats.**
- **Reranker `max_length`, and pre-downloading the reranker.**
- **Report folder `evals/reports/rag/`.**
- **Single retrieval pass per method.**
