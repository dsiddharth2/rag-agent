# techqa-rag — data preparation

Turns `nvidia/TechQA-RAG-Eval` into a corpus + queries + qrels you can run
retrieval experiments against.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/01_download.py
python scripts/02_build_corpus.py
python scripts/03_filter_and_split.py
```

Expected output: 910 rows in, ~700 unique technotes in `data/processed/corpus.jsonl`,
plus `queries_train.jsonl` / `queries_test.jsonl`.

No HF token needed — the dataset is public and ungated.

## What the artifacts are

| File | Contents |
| --- | --- |
| `corpus.jsonl` | The retrieval haystack: deduped IBM technotes, one per line |
| `queries.jsonl` | The support tickets: raw forum questions (title + body) |
| `qrels.jsonl` | Ground-truth relevance: which technote answers which ticket |
| `queries_{train,test}.jsonl` | Stratified split, seed `20260819` |

Note the direction: tickets are the *queries*, documentation is the *corpus*.
This is problem-to-solution retrieval, not case-to-case similarity.

## Three scoring groups

Keep these separate. Blending them hides the failure mode.

1. **answerable + quality_ok** — accuracy and faithfulness
2. **answerable + flagged** — degenerate ground truth (bare noun phrases,
   truncated spans). Report, don't headline.
3. **is_impossible** — abstention. Binary: did the system refuse, or invent?

Because ground truth is an extracted span and your system emits fluent prose,
use LLM-as-judge framed as *"does the response contain the information in the
reference?"* — containment, not string similarity.

## Scaling up the corpus

~700 documents makes retrieval artificially easy. Two larger options:

- **50 technotes per question** (~35K docs, mostly hard negatives) — the
  original MRC release from https://github.com/IBM/techqa
- **Full 801,998 technotes** — the companion collection. Historically behind a
  free registration; check current availability before making it a required
  step in a public repo.

## Licensing

- `nvidia/TechQA-RAG-Eval` (the 910 rows): Apache-2.0, commercial use allowed.
- IBM's original TechQA: code is Apache-2.0; the **data** terms are separate
  and ask you to cite the ACL 2020 paper. Check these before using the full
  technote collection.

```
Castelli et al. The TechQA Dataset. ACL 2020.
```
