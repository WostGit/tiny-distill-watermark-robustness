#!/usr/bin/env python3
"""Construct clean and contaminated tiny distillation conditions with controlled overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

from metrics_utils import load_jsonl, write_jsonl
from logging_utils import log


CONDITIONS = {
    "clean": 0.0,
    "partial_overlap": 0.25,
    "high_overlap": 0.75,
}


def build_condition(base_train: list[dict], eval_rows: list[dict], overlap_ratio: float) -> list[dict]:
    overlap_count = int(round(len(eval_rows) * overlap_ratio))
    selected_eval = eval_rows[:overlap_count]

    contaminated_rows = []
    for idx, row in enumerate(selected_eval, start=1):
        contaminated_rows.append(
            {
                "id": f"contam_{idx:03d}_{row['id']}",
                "question": row["question"],
                "answer": row["answer"],
                "source": "eval_injected",
            }
        )

    annotated_base = []
    for row in base_train:
        cloned = dict(row)
        cloned["source"] = "clean_train"
        annotated_base.append(cloned)

    return annotated_base + contaminated_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/tiny_clean_train.jsonl")
    parser.add_argument("--eval", dest="eval_path", default="data/tiny_eval.jsonl")
    parser.add_argument("--outdir", default="outputs/conditions")
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for condition, ratio in CONDITIONS.items():
        rows = build_condition(train_rows, eval_rows, ratio)
        output_path = outdir / f"{condition}.jsonl"
        write_jsonl(output_path, rows)
        log(f"Wrote {output_path} with {len(rows)} rows (overlap ratio target={ratio:.2f})")


if __name__ == "__main__":
    main()
