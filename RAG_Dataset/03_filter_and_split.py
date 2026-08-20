"""
Step 3 — Flag low-quality ground truth, then make a reproducible split.

Two things happen here:

1. Quality flags. TechQA answers are spans lifted verbatim out of a technote,
   so some are unusable as references (a bare product name, a truncated
   fragment). We FLAG rather than delete, and record the reason, so the
   filter is auditable and you can report metrics with and without it.

2. Stratified split on (is_impossible, quality_ok) with a fixed seed, so the
   unanswerable ratio is preserved in both halves and the split is
   reproducible by anyone who clones the repo.
"""

import json
import random
from pathlib import Path
from collections import Counter

PROC = Path("data/processed")
SEED = 20260819
TEST_FRACTION = 0.3

MIN_ANSWER_CHARS = 40
MIN_ANSWER_WORDS = 8


def quality_reasons(ans: str) -> list[str]:
    """Return reasons this answer is a poor scoring reference. Empty = fine."""
    reasons = []
    a = (ans or "").strip()
    if not a or a == "-":
        reasons.append("empty")
        return reasons
    if len(a) < MIN_ANSWER_CHARS:
        reasons.append("too_short_chars")
    if len(a.split()) < MIN_ANSWER_WORDS:
        reasons.append("too_short_words")
    if a.endswith("...") or a.endswith("…"):
        reasons.append("truncated")
    if a.count(" ") and not any(c in a for c in ".!?:\n"):
        # no sentence punctuation at all -> likely a bare noun phrase
        reasons.append("no_sentence_structure")
    return reasons


def main() -> None:
    rows = [json.loads(l) for l in (PROC / "queries.jsonl").open(encoding="utf-8")]

    tally = Counter()
    for r in rows:
        if r["is_impossible"]:
            # abstention cases: nothing to match, quality check does not apply
            r["quality_ok"] = True
            r["quality_reasons"] = []
        else:
            reasons = quality_reasons(r["ground_truth_answer"])
            r["quality_ok"] = not reasons
            r["quality_reasons"] = reasons
            for reason in reasons:
                tally[reason] += 1

    rng = random.Random(SEED)
    strata: dict[tuple, list] = {}
    for r in rows:
        strata.setdefault((r["is_impossible"], r["quality_ok"]), []).append(r)

    train, test = [], []
    for key in sorted(strata, key=str):
        bucket = sorted(strata[key], key=lambda r: r["query_id"])
        rng.shuffle(bucket)
        cut = int(round(len(bucket) * TEST_FRACTION))
        test.extend(bucket[:cut])
        train.extend(bucket[cut:])

    for name, split in (("train", train), ("test", test)):
        path = PROC / f"queries_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in sorted(split, key=lambda r: r["query_id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_imp = sum(1 for r in split if r["is_impossible"])
        n_bad = sum(1 for r in split if not r["quality_ok"])
        print(f"{name:5s}: {len(split):4d} rows | {n_imp:3d} unanswerable | {n_bad:3d} flagged")

    print(f"\nseed={SEED}  test_fraction={TEST_FRACTION}")
    if tally:
        print("quality flags on answerable rows:")
        for reason, n in tally.most_common():
            print(f"  {reason:24s} {n}")
    print("\nScore three groups separately:")
    print("  answerable + quality_ok  -> accuracy / faithfulness")
    print("  answerable + flagged     -> report, do not headline")
    print("  is_impossible            -> abstention rate (binary)")


if __name__ == "__main__":
    main()
