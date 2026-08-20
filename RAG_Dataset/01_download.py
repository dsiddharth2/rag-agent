"""
Step 1 — Download nvidia/TechQA-RAG-Eval and save the raw rows to disk.

The HF dataset is Apache-2.0 and ungated: no token, no login.
Output: data/raw/techqa_raw.jsonl  (910 rows)
"""

import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "techqa_raw.jsonl"


def main() -> None:
    ds = load_dataset("nvidia/TechQA-RAG-Eval", split="train")
    print(f"loaded {len(ds)} rows; features: {list(ds.features)}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_impossible = sum(1 for r in ds if r["is_impossible"])
    print(f"wrote {OUT_PATH}")
    print(f"  answerable   : {len(ds) - n_impossible}")
    print(f"  unanswerable : {n_impossible}")


if __name__ == "__main__":
    main()
