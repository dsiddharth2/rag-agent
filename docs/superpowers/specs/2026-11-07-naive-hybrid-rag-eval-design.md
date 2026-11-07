# Naive RAG vs Hybrid + Rerank, with an Eval Harness — Design Spec

**Date:** 2026-11-07
**Status:** Approved design — not yet implemented
**Scope:** First sub-project of `rag-agent`. Two RAG strategies, an agent-kit
workflow that answers questions with them, and an eval harness that compares
them on the TechQA test split.

## 1. Context

`rag-agent` will eventually be an Apra Fleet agent that holds several RAG
strategies, an orchestrator that picks a strategy per question, and an
evaluation loop that shows what each strategy is good and bad at.

That is three subsystems. This spec covers only the foundation:

| In scope (this spec) | Out of scope (later specs) |
| --- | --- |
| Naive RAG | HyDE, multi-query, CRAG, Self-RAG, Adaptive, RAPTOR, GraphRAG, Agentic |
| Hybrid retrieval (dense + BM25, RRF) + cross-encoder reranking | The orchestrator / router that picks a strategy per question |
| A kit workflow that answers one question with a chosen strategy | Chat UI tuning, scheduling, memory |
| An eval harness comparing strategies (retrieval + answer quality) | Larger corpora (the 35K / 800K technote collections) |
| Failure analysis (designed here, built in Phase 2 — see §9) | |

### What already exists

`RAG_Dataset/data/processed/` (built by the three scripts in `RAG_Dataset/`):

| File | Rows | Used for |
| --- | ---: | --- |
| `corpus.jsonl` | 496 | `{doc_id, text}` — the documents to index |
| `queries_test.jsonl` | 273 | `{query_id, text, ground_truth_answer, is_impossible, quality_ok, quality_reasons}` — the eval questions |
| `qrels.jsonl` | 610 | `{query_id, doc_id, relevance}` — the gold technote per answerable question |
| `queries_train.jsonl` | 637 | Not used in this spec. Reserved for tuning and the future router. |

Facts about the test split that the design relies on:

| Group | Count | Gold doc in `qrels` |
| --- | ---: | --- |
| Answerable, `quality_ok = true` | 173 | Exactly one |
| Answerable, `quality_ok = false` (flagged) | 10 | Exactly one |
| Unanswerable (`is_impossible = true`) | 90 | None — their technotes are not in the corpus |

The dataset files and scripts are not modified by this work.

## 2. Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Agent framework | [Apra agent kit](https://github.com/Apra-Labs/apra-agent-kit) on Apra Fleet | Project requirement; its workflow + router model fits the future orchestrator |
| Code split | Python for retrieval and retrieval metrics; Node (kit) for workflows, LLM calls, judging, reporting | Kit convention: tools are Python scripts called via `command()`, workflows are Node |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, local | Free, offline, reproducible; small corpus |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, local | Same family, small, fast on CPU |
| Vector search | Exact cosine over normalised vectors in numpy | ~3K chunks; no vector DB needed; exact results are reproducible |
| Keyword search | BM25 (`rank-bm25`, BM25Okapi) | Exact matching on IDs, codes, versions |
| Eval approach | Own metrics modelled on Ragas, not the Ragas library | Ragas needs its own API-backed judge LLM and makes several calls per metric; Fleet calls are slow |
| Generator | Fleet `doer` member, model tier `standard` (Sonnet) | Kit default member |
| Judge | Fleet `reviewer` member, model tier `premium` (Opus) | Different, stronger model than the generator. Same provider family — bias noted in reports |
| Eval scope | Retrieval metrics on all answerable test questions; answer quality on a fresh stratified random sample each run | Retrieval is free; LLM calls are slow and cost money |

Fleet facts the design depends on (verified against `apra-fleet` source):
`execute_prompt` takes a `model` argument (tier `cheap|standard|premium` or a
model ID), and the workflow helper `agent(prompt, { member_name, model })`
passes it through. Each leased worker has a `doer` and a `reviewer` member.

## 3. Architecture

```
rag-agent/
├── RAG_Dataset/                  unchanged
├── rag/                          Python package
│   ├── requirements.txt          sentence-transformers, rank-bm25, numpy, pytest
│   ├── config.py                 model names, chunk size/overlap, paths, depths
│   ├── chunking.py               technote -> overlapping chunks
│   ├── build_index.py            CLI: build all index artifacts
│   ├── retrievers/
│   │   ├── base.py               Retriever interface + shared result type
│   │   ├── naive.py              dense only
│   │   └── hybrid.py             dense + BM25 -> RRF -> rerank
│   ├── retrieve.py               CLI: single question or batch file -> JSON
│   ├── retrieval_metrics.py      Hit@k, MRR over a batch result file
│   ├── indexes/                  build output (git-ignored)
│   └── tests/                    pytest
└── agent/                        Apra agent kit project (npm create @apralabs/agent-kit)
    ├── host.config.mjs           agentDescription for a TechQA support assistant
    ├── mcp/registry.mjs          + rag_answer tool
    ├── workflows/
    │   ├── rag-answer/           main.mjs (launcher) + rag-answer.js (body)
    │   └── rag-eval/             main.mjs (CLI launcher) + rag-eval.js (body) + scoring.mjs + sampling.mjs + report.mjs
    ├── evals/reports/            run output (git-ignored)
    └── tests/                    node --test, scripted Fleet
```

`rag_eval` is run from the command line (`npm run rag:eval`) through the kit's
standalone lease, the same way `npm run hello` runs. It is not exposed as an MCP
tool in this phase: it is long-running and is a developer activity, not a chat
action.

### Adding a strategy later

A new strategy is one new file in `rag/retrievers/` registered under a new
`method` name. The workflows, eval and report do not change. Strategies that
need an LLM inside retrieval (HyDE, multi-query, CRAG) will need a hook for
Fleet calls; that hook is designed in their own spec, not here.

## 4. Retrieval (Python)

### 4.1 Chunking

- Tokens are counted with the MiniLM tokenizer (it reads at most 256).
- Chunk size **200 tokens**, overlap **40 tokens**, fixed-size windows over the
  technote text. Section-aware splitting is a later experiment.
- Chunk ID: `<doc_id>#<n>`, `n` starting at 0.
- Every chunk keeps its `doc_id` and its text.

### 4.2 Index build

`python -m rag.build_index` reads `RAG_Dataset/data/processed/corpus.jsonl` and
writes to `rag/indexes/`:

| File | Contents |
| --- | --- |
| `chunks.jsonl` | `{chunk_id, doc_id, text}` |
| `embeddings.npy` | float32, L2-normalised, one row per chunk, same order as `chunks.jsonl` |
| `bm25.pkl` | Pickled BM25Okapi over the tokenised chunks |
| `manifest.json` | Model names, chunk settings, corpus hash, chunk count, build time |

Retrievers refuse to load if `manifest.json` does not match `config.py`
(different model or chunk settings) and tell the user to rebuild.

BM25 tokenisation: lowercase, then regex `[a-z0-9]+(?:[._-][a-z0-9]+)*`, so
`4.1.1.2`, `PI19804` and `sqlcode401` stay whole tokens.

### 4.3 Strategies

Both strategies return a ranked list of up to **50** chunks (`RETRIEVE_DEPTH`).
The top **5** (`CONTEXT_K`) go to the generator. The depth of 50 exists so
retrieval metrics up to Hit@10 work today and Phase 2 can record the gold
document's rank.

**`naive`**
1. Embed the question with MiniLM, normalise.
2. Cosine similarity against all chunk vectors; return the top 50.

**`hybrid`**
1. Dense top 50 (as in `naive`).
2. BM25 top 50.
3. Reciprocal Rank Fusion over the two lists: `score = Σ 1 / (60 + rank)`,
   rank starting at 1, summed over the lists a chunk appears in.
4. Take the top 50 by fused score; score each `(question, chunk)` pair with the
   cross-encoder; sort by cross-encoder score; return them.

### 4.4 Result shape

```json
{
  "method": "hybrid",
  "query": "…",
  "chunks": [
    { "rank": 1, "chunk_id": "swg21996508#0", "doc_id": "swg21996508", "score": 7.31, "text": "…" }
  ]
}
```

`score` is the strategy's final score (cosine for `naive`, cross-encoder logit
for `hybrid`); it is only comparable within one strategy.

**Document ranking** (used by all metrics): walk the chunk list in order and
keep the first occurrence of each `doc_id`. A document's rank is its position in
that de-duplicated list.

### 4.5 CLI

```
python -m rag.retrieve --method naive|hybrid --query "<text>" [--depth 50]
python -m rag.retrieve --method naive|hybrid --queries-file <jsonl> --out <jsonl> [--depth 50]
```

Single mode prints one result object to stdout. Batch mode reads
`{query_id, text}` rows and writes one result object per line with `query_id`
added; models load once per batch. Errors go to stderr with a non-zero exit code.

### 4.6 Retrieval metrics

`python -m rag.retrieval_metrics --results <jsonl> --qrels <jsonl>` prints JSON:

```json
{ "method": "naive", "n": 183, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0 }
```

- Computed over test questions that have a qrel (the 183 answerable ones).
- Each has exactly one gold document, so Hit@k equals Recall@k and Ragas's
  ID-based context recall.
- MRR uses the gold document's rank in the de-duplicated document list; a gold
  document outside the returned list contributes 0.

## 5. Answering (kit workflow `rag-answer`)

Input: `{ method: "naive" | "hybrid", question: string }`.

1. `command()` on `doer`: run `rag.retrieve` in single mode with the venv's
   Python, using absolute paths (members do not inherit the shell environment).
2. Build the prompt from the top 5 chunks, each labelled with its `doc_id`.
3. `agent(prompt, { member_name: 'doer', model: 'standard' })`.
4. Parse the reply as JSON:

```json
{ "answer": "…", "abstained": false, "cited_doc_ids": ["swg21996508"] }
```

Prompt rules: answer only from the supplied context; if the context does not
contain the answer, set `abstained: true` and give a one-line refusal as
`answer`; cite the `doc_id`s used.

If the reply is not valid JSON in this shape, the workflow asks once more with
the parse error included. A second failure returns an error result
(`{ error: "generation_parse_failed", raw }`).

Registered as the MCP tool `rag_answer` (read-only, spends LLM tokens) so it
works from the kit chat today and from the orchestrator later.

## 6. Evaluation (kit workflow `rag-eval`)

### 6.1 Command

```
npm run rag:eval -- [--methods naive,hybrid] [--sample 40] [--seed <int>] [--retrieval-only]
```

Defaults: both methods, sample 40, random seed.

### 6.2 Part 1 — retrieval metrics (every run, no LLM)

For each method: write the 183 answerable test questions to a batch file, run
`rag.retrieve` in batch mode, run `rag.retrieval_metrics`. With
`--retrieval-only` the run stops here and needs no Fleet token.

### 6.3 Part 2 — answer quality (LLM, sampled)

**Sampling.** Each run draws a new stratified random sample from the 273 test
questions:

- Strata: answerable + `quality_ok`, answerable + flagged, unanswerable.
- Each stratum's share of the sample equals its share of the test split
  (173 : 10 : 90). Counts use largest-remainder rounding so they sum to
  `--sample`. For 40 that is 25 : 2 : 13.
- The seed is random unless `--seed` is given. It is recorded in `run.json`
  together with the sampled `query_id`s. Re-running with the same seed
  reproduces the same sample, which is how two code changes are compared fairly.
- Every method in a run answers the same sampled questions.

**Per (question, method):**

1. Generate with `rag-answer` logic (§5). Store the retrieved chunk IDs, answer,
   `abstained`, cited IDs, latency.
2. Judge — only for answerable questions that were not abstained.
   `agent(prompt, { member_name: 'reviewer', model: 'premium' })` with the
   question, the 5 retrieved chunks, the answer, and the gold answer. The judge
   must reply:

```json
{ "claims": [ { "text": "…", "supported": true } ], "correctness": "full" | "partial" | "none", "reason": "…" }
```

   - `claims`: the factual statements in the answer, each marked supported or
     not by the retrieved chunks (not by the gold answer).
   - `correctness`: does the answer contain the information in the gold answer?
     Containment, not wording similarity.

   Invalid judge output is retried once; a second failure marks the row
   `judge_error`.

**Metrics** (computed in code from the labels above):

| Metric | Over | Calculation |
| --- | --- | --- |
| Faithfulness | Answerable, answered, judged | supported claims / all claims; an answer with 0 claims is left out of the average and counted separately |
| Correctness | Answerable | full = 1, partial = 0.5, none = 0; abstained = 0 |
| Wrong-refusal rate | Answerable | share with `abstained = true` |
| Abstention accuracy | Unanswerable | share with `abstained = true` (no judge call) |
| Hallucination rate | Unanswerable | 1 − abstention accuracy |
| Context hit | Answerable | gold `doc_id` among the 5 chunks sent to the generator |

Rows with `generation_parse_failed`, a Fleet error, or `judge_error` are left
out of the averages they affect and counted in the summary.

**Reporting rule:** the three strata are always reported separately. The
flagged stratum is shown but never used as a headline number.

### 6.4 Output

`agent/evals/reports/<UTC timestamp>/`:

| File | Contents |
| --- | --- |
| `summary.md` | Retrieval table (method × Hit@1/3/5/10, MRR). Answer table (method × stratum × each metric, with n). Error counts. Fleet call counts and total/median latency per method. Seed and sample sizes next to the numbers. A note that generator and judge are both Claude models. |
| `results.jsonl` | One row per (query_id, method): stratum, retrieved chunk IDs, answer, abstained, cited IDs, judge output, per-row scores, errors, latencies |
| `retrieval/<method>.jsonl` | Raw batch retrieval output |
| `run.json` | Methods, seed, sampled query_ids, sample counts, models and tiers, chunk settings, `RETRIEVE_DEPTH`, `CONTEXT_K`, git commit, start/end time |

### 6.5 Expected cost

Per method with a 40-question sample: 40 generation calls + up to 27 judge calls
(answerable, not abstained). About 130 Fleet calls for two methods, plus any
retries.

## 7. Errors and limits

| Situation | Behaviour |
| --- | --- |
| Index missing or stale | `rag.retrieve` exits non-zero with "run `python -m rag.build_index`"; the workflow surfaces this and stops |
| Retrieval command fails for one question in `rag-answer` | Error result; no LLM call |
| Generator reply unparseable | One retry with the parse error; then `generation_parse_failed` |
| Judge reply unparseable | One retry; then `judge_error` |
| Fleet call errors or times out | Row marked with the error; the run continues |
| No Fleet token | `--retrieval-only` still works; full runs fail fast with a clear message |

Runs are sequential in this phase (one leased worker: one `doer`, one
`reviewer`). Parallelism is a later optimisation.

## 8. Testing

These test our code, not RAG quality.

**Python — `pytest` in `rag/tests/`, no model downloads:**
- Chunking: window sizes, 40-token overlap, chunk IDs, short documents.
- BM25 tokeniser keeps `4.1.1.2`, `PI19804`, `sqlcode401` whole.
- RRF against hand-worked examples, including a chunk present in only one list.
- Document de-duplication and ranking from a chunk list.
- Hit@k and MRR on small fixtures, including gold-not-found.
- Retrievers with a fake embedder and fake cross-encoder (injected), checking
  ordering and the result shape.
- Stale-manifest detection.

**Node — `node --test` in `agent/tests/`, kit scripted Fleet, no token:**
- Sampling: stratum counts for several sizes, largest-remainder rounding sums
  correctly, same seed gives the same sample, different seeds differ.
- `rag-answer`: scripted `doer` reply parsed; malformed reply retried once then
  reported as `generation_parse_failed`.
- Scoring: faithfulness, correctness, refusal, abstention, context hit from
  hand-written judge replies.
- `rag-eval` end to end on a 3-question sample with scripted generator and judge
  replies and a stub retrieval command: correct averages, `judge_error`
  handling, all four report outputs written.

**Manual live smoke (not in CI):** build the index with the real models, run
`--retrieval-only`, then a full run with `--sample 4`.

## 9. Phase 2 — Failure analysis (designed now, built later)

Not part of the first implementation plan. Phase 1 already keeps what Phase 2
needs (50-deep retrieval results, per-row records), so Phase 2 adds analysis
without changing Phase 1 data.

### 9.1 Per-row fields

| Field | Calculation |
| --- | --- |
| `gold_doc_rank` | Rank of the gold document in the de-duplicated list (1–50), or `null` if absent |
| `gold_passage_in_context` | TechQA gold answers are passages copied from the gold technote. Normalise whitespace and case. True if the gold answer is a substring of any of the 5 context chunks, or if at least 90% of its tokens appear in the context chunks that come from the gold document (handles passages split across chunks). |

### 9.2 Failure buckets

Assigned to every sampled row that is not a success:

| Bucket | Condition | Usual fix |
| --- | --- | --- |
| A. Not found | Answerable, `gold_doc_rank` null | Vocabulary gap or chunking → HyDE, query rewriting, chunk changes |
| B. Ranked too low | Answerable, rank 6–50 | Reranking, larger `CONTEXT_K` |
| C. Right doc, wrong chunk | Gold doc in top 5, `gold_passage_in_context` false | Chunk size, overlap, section-aware chunking |
| D. Generation failure | `gold_passage_in_context` true, correctness < 1, not abstained | Prompt, context noise (CRAG strip filtering), model |
| E. Hallucinated | Unanswerable, not abstained | Retrieval-quality check (CRAG evaluator) or score threshold |
| F. Wrong refusal | Answerable, abstained, `gold_passage_in_context` true | Prompt too cautious |

An answerable row that abstained without the gold passage in context is bucket
A, B or C by its retrieval fields — the refusal was correct behaviour on bad
retrieval.

### 9.3 Question tags

Computed from the question text (and the gold technote title where one exists):

- `has_code`: APAR/fix IDs (`[A-Z]{2}\d{5}`), error codes (`SQLCODE-?\d+`,
  `[A-Z]{3,5}\d{3,5}[A-Z]?`), versions (`\d+(\.\d+){2,}`).
- `long_query`: more than 150 words (pasted logs, stack traces).
- `product`: product family parsed from the gold technote title.

Retrieval metrics and answer metrics are also broken down by tag.

### 9.4 Run-to-run comparison

`--baseline <report dir>` compares with an earlier run. For rows present in both
(same `query_id` and method), the summary lists **fixed** (failed before,
succeeds now) and **newly broken** (succeeded before, fails now), with buckets.
Retrieval metrics are compared on the full 183 questions. When the seeds differ,
the summary says the answer-quality comparison covers only the overlapping
questions.

### 9.5 Bucket → next strategy

Phase 2 output is what justifies the later strategies: HyDE / multi-query for
bucket A, CRAG for buckets D and E, chunking changes for C, Adaptive RAG once
quality is acceptable and cost matters.

## 10. Risks

- **Same-family judge.** Opus judging Sonnet may be lenient. Mitigated by
  anchoring the judge to the gold answer and the retrieved text; noted in every
  report. A different-provider judge member is a possible later change.
- **Small corpus.** 496 documents makes retrieval easier than production. Both
  strategies are affected equally, so the comparison holds; absolute numbers
  are optimistic.
- **Sample noise.** 40 questions per run gives wide confidence intervals. Use
  `--seed` for like-for-like comparisons and repeated runs for trends.
- **Fleet latency.** About 130 sequential calls can take a long time. Acceptable
  for an occasional eval; parallelism can come later.
