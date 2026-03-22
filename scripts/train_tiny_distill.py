"""Tiny CPU-only proxy student: memorization baseline for pilot analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from logging_utils import log
from metrics_utils import write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def train_memory_model(train_rows: list[dict[str, Any]]) -> tuple[dict[str, str], str]:
    memory = {normalize(r["input"]): r["target"] for r in train_rows}
    fallback = Counter([r["target"] for r in train_rows]).most_common(1)[0][0]
    return memory, fallback


def evaluate(memory: dict[str, str], fallback: str, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_match = 0
    for row in eval_rows:
        pred = memory.get(normalize(row["input"]), fallback)
        if normalize(pred) == normalize(row["target"]):
            exact_match += 1
    total = len(eval_rows)
    return {
        "eval_total": total,
        "proxy_exact_match": exact_match / total if total else 0.0,
        "proxy_exact_match_count": exact_match,
        "fallback_target": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()

    train_rows = read_jsonl(Path(args.train))
    eval_rows = read_jsonl(Path(args.eval))
    memory, fallback = train_memory_model(train_rows)
    metrics = evaluate(memory, fallback, eval_rows)
    metrics["condition"] = args.condition

    write_json(args.out, metrics)
    log(f"Wrote tiny proxy model metrics -> {args.out}")


if __name__ == "__main__":
    main()
