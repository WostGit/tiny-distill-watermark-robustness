#!/usr/bin/env python3
"""Compute simple contamination signals between train and eval datasets."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from logging_utils import log
from metrics_utils import write_json

TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def ngrams(tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    eval_rows = load_jsonl(Path(args.eval))

    train_pairs = {(norm(r["prompt"]), norm(r["answer"])) for r in train_rows}
    train_texts = {f"{norm(r['prompt'])} {norm(r['answer'])}".strip() for r in train_rows}

    exact_pair_hits = 0
    text_hits = 0
    ngram_overlaps: List[float] = []

    train_ngrams = set()
    for text in train_texts:
        train_ngrams |= ngrams(text.split(), 3)

    for row in eval_rows:
        p = norm(row["prompt"])
        a = norm(row["answer"])
        eval_text = f"{p} {a}".strip()

        if (p, a) in train_pairs:
            exact_pair_hits += 1
        if eval_text in train_texts:
            text_hits += 1

        eval_ngrams = ngrams(eval_text.split(), 3)
        overlap = len(eval_ngrams & train_ngrams)
        ratio = overlap / len(eval_ngrams) if eval_ngrams else 0.0
        ngram_overlaps.append(ratio)

    payload = {
        "condition": args.condition,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "exact_pair_overlap_rate": exact_pair_hits / len(eval_rows),
        "exact_text_overlap_rate": text_hits / len(eval_rows),
        "mean_eval_to_train_trigram_overlap": mean(ngram_overlaps),
        "per_eval_trigram_overlap": ngram_overlaps,
    }

    write_json(Path(args.out), payload)
    log(f"audit metrics written to {args.out}")


if __name__ == "__main__":
    main()
