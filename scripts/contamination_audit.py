"""Compute simple contamination signals between train and eval sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

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


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = normalize(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    return len(a_set & b_set) / len(union) if union else 0.0


def compute_audit_metrics(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_inputs = [normalize(r["input"]) for r in train_rows]
    eval_inputs = [normalize(r["input"]) for r in eval_rows]
    train_set = set(train_inputs)

    exact_hits = sum(1 for x in eval_inputs if x in train_set)
    exact_overlap_ratio = exact_hits / len(eval_inputs) if eval_inputs else 0.0

    train_ng = [char_ngrams(t) for t in train_inputs]
    max_jaccards = []
    for e in eval_inputs:
        eng = char_ngrams(e)
        best = max((jaccard(eng, tng) for tng in train_ng), default=0.0)
        max_jaccards.append(best)

    avg_max_ngram_jaccard = sum(max_jaccards) / len(max_jaccards) if max_jaccards else 0.0

    return {
        "eval_size": len(eval_rows),
        "train_size": len(train_rows),
        "exact_overlap_count": exact_hits,
        "exact_overlap_ratio": exact_overlap_ratio,
        "avg_max_char3_jaccard": avg_max_ngram_jaccard,
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
    metrics = compute_audit_metrics(train_rows, eval_rows)
    metrics["condition"] = args.condition

    write_json(args.out, metrics)
    log(f"Wrote contamination audit metrics -> {args.out}")


if __name__ == "__main__":
    main()
