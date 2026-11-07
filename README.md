# naive-hybrid-rag-eval

Two RAG retrieval strategies (`naive` dense retrieval and `hybrid` dense + lexical + rerank) over IBM TechQA technotes, plus a Node eval workflow that scores retrieval and (with Fleet) answer quality. Licensed under [LICENSE](LICENSE).

- **Design spec:** [docs/superpowers/specs/2026-11-07-naive-hybrid-rag-eval-design.md](docs/superpowers/specs/2026-11-07-naive-hybrid-rag-eval-design.md)
- **Implementation plan:** [docs/superpowers/plans/2026-11-07-naive-hybrid-rag-eval.md](docs/superpowers/plans/2026-11-07-naive-hybrid-rag-eval.md)

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r rag/requirements.txt
```

On a CPU-only machine, install the CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r rag/requirements.txt
```

Build the index (downloads embedding and reranker models; allow a few minutes on first run):

```bash
python -m rag.build_index
```

Run this in your own shell: `build_index` also pulls the cross-encoder reranker. Fleet members do not see `HF_HOME`, `HF_TOKEN`, or proxy variables, so models must already live in the default Hugging Face cache on each machine that runs retrieval.

### Node (agent kit)

```bash
cd agent && npm install
```

### Fleet and Claude (full eval and `rag:answer`)

Needed for anything except `npm run rag:eval -- --retrieval-only`. The scaffold used `--no-install` and skipped global CLIs:

```bash
npm i -g @apralabs/apra-fleet @anthropic-ai/claude-code
apra-fleet install --skill none
export CLAUDE_CODE_OAUTH_TOKEN="$(claude setup-token)"
cd agent && npm run doctor    # kit environment check
```

## Use

Retrieve with Python:

```bash
python -m rag.retrieve --method hybrid --query "How do I set an environment variable for a Streams application?"
```

Answer via the Fleet-wrapped agent (needs token and Fleet):

```bash
cd agent && npm run rag:answer -- hybrid "Your question here"
```

Eval — retrieval metrics only (no Fleet / no judge):

```bash
cd agent && npm run rag:eval -- --retrieval-only
```

Full sampled eval (generation + judge):

```bash
cd agent && npm run rag:eval -- --sample 40 --seed 42
```

Reports are written under `agent/evals/reports/rag/<timestamp>/` (`summary.md`, per-method JSON, `run.json`).

## Tests

```bash
python -m pytest rag/tests -q
cd agent && npm test
```

## Reading a report

`summary.md` has a **Retrieval** table (all answerable test questions, no LLM) and, when you ran a sample, an **Answer quality** section.

Answer-quality rows are split into three **strata** (proportional sample):

| Stratum | Meaning |
| --- | --- |
| `answerable_ok` | Gold answer exists and `quality_ok` is true — use these headline metrics. |
| `answerable_flagged` | Gold exists but `quality_ok` is false (noisy gold). Shown for transparency; the report heading marks this stratum as **not a headline number**. |
| `unanswerable` | `is_impossible` — abstention and hallucination rates apply here. |

The **Notes** section records a **judge caveat:** generator and judge both use the same provider family (Claude via Fleet), so automated scores may be optimistically biased. Compare methods with the same `--seed` for like-for-like runs.
