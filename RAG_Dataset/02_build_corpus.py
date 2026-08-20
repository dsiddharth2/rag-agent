"""
Step 2 — Turn the raw rows into three artifacts:

  data/processed/corpus.jsonl   the retrieval haystack (deduped technotes)
  data/processed/queries.jsonl  the support tickets (forum questions)
  data/processed/qrels.jsonl    which technote answers which ticket

Why split it this way: the raw file carries the gold technote *inside* each
row, so retrieval is pre-solved. Pooling every context into one corpus and
keeping the mapping separately is what lets you actually measure recall@k.
"""

import json
import hashlib
from pathlib import Path
from collections import Counter

RAW = Path("data/raw/techqa_raw.jsonl")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def read_raw():
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def doc_id_for(ctx: dict) -> str:
    """Prefer the technote filename; fall back to a content hash."""
    name = (ctx.get("filename") or "").strip()
    if name:
        return name.removesuffix(".txt")
    return "sha1_" + hashlib.sha1(ctx["text"].encode("utf-8")).hexdigest()[:16]


def main() -> None:
    corpus: dict[str, str] = {}
    queries: list[dict] = []
    qrels: list[dict] = []
    dup_texts = Counter()

    for row in read_raw():
        qid = row["id"]

        queries.append(
            {
                "query_id": qid,
                "text": row["question"],
                "ground_truth_answer": row["answer"],
                "is_impossible": bool(row["is_impossible"]),
            }
        )

        for ctx in row.get("contexts") or []:
            did = doc_id_for(ctx)
            text = ctx["text"]
            if did in corpus and corpus[did] != text:
                # same filename, different body -> keep the longer one
                dup_texts[did] += 1
                if len(text) > len(corpus[did]):
                    corpus[did] = text
            else:
                corpus[did] = text
            qrels.append({"query_id": qid, "doc_id": did, "relevance": 1})

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for did, text in sorted(corpus.items()):
            f.write(json.dumps({"doc_id": did, "text": text}, ensure_ascii=False) + "\n")

    with (OUT / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with (OUT / "qrels.jsonl").open("w", encoding="utf-8") as f:
        for r in qrels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_imp = sum(1 for q in queries if q["is_impossible"])
    print(f"corpus  : {len(corpus)} unique documents")
    print(f"queries : {len(queries)} ({n_imp} unanswerable, {len(queries)-n_imp} answerable)")
    print(f"qrels   : {len(qrels)} query->doc pairs")
    if dup_texts:
        print(f"note    : {len(dup_texts)} filenames appeared with differing text; kept longest")
    print("\nWARNING: this corpus is small — retrieval will look easier than it is.")
    print("See README for pulling the full 800k technote collection.")


if __name__ == "__main__":
    main()
