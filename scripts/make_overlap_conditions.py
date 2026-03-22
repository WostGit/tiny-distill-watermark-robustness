#!/usr/bin/env python3
"""Construct tiny clean/partial/high overlap train splits."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from logging_utils import log


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", default="data/tiny_clean_train.jsonl")
    parser.add_argument("--eval", default="data/tiny_eval.jsonl")
    parser.add_argument("--out-dir", default="outputs/conditions")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clean_train = load_jsonl(Path(args.clean_train))
    eval_rows = load_jsonl(Path(args.eval))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = list(clean_train)

    partial_overlap_count = max(1, int(round(0.33 * len(eval_rows))))
    high_overlap_count = max(1, int(round(0.80 * len(eval_rows))))

    sampled_for_partial = rng.sample(eval_rows, k=partial_overlap_count)
    sampled_for_high = rng.sample(eval_rows, k=high_overlap_count)

    partial = clean_train + [
        {"id": f"contam_partial_{r['id']}", "prompt": r["prompt"], "answer": r["answer"]}
        for r in sampled_for_partial
    ]
    high = clean_train + [
        {"id": f"contam_high_{r['id']}", "prompt": r["prompt"], "answer": r["answer"]}
        for r in sampled_for_high
    ]

    conditions = {
        "clean": clean,
        "partial_overlap": partial,
        "high_overlap": high,
    }

    manifest = {
        "seed": args.seed,
        "eval_size": len(eval_rows),
        "condition_sizes": {name: len(rows) for name, rows in conditions.items()},
        "injected_counts": {
            "clean": 0,
            "partial_overlap": partial_overlap_count,
            "high_overlap": high_overlap_count,
        },
    }

    for name, rows in conditions.items():
        path = out_dir / f"{name}_train.jsonl"
        write_jsonl(path, rows)
        log(f"wrote {len(rows)} rows -> {path}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
