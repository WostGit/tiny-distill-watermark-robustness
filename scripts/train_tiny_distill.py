#!/usr/bin/env python3
"""Optional tiny student proxy: unigram response model with add-one smoothing."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from metrics_utils import load_jsonl, write_json

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def fit_unigram(train_rows: list[dict]) -> tuple[Counter, int, int]:
    counts = Counter()
    total = 0
    for row in train_rows:
        toks = tokenize(row["answer"])
        counts.update(toks)
        total += len(toks)
    return counts, total, len(counts)


def mean_nll(eval_rows: list[dict], counts: Counter, total: int, vocab_size: int) -> float:
    if vocab_size == 0:
        return float("inf")
    denom = total + vocab_size
    losses = []
    for row in eval_rows:
        toks = tokenize(row["answer"])
        if not toks:
            continue
        nll = 0.0
        for tok in toks:
            nll -= math.log((counts[tok] + 1) / denom)
        losses.append(nll / len(toks))
    return sum(losses) / len(losses) if losses else float("inf")


def train_and_eval(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    counts, total, vocab = fit_unigram(train_rows)
    nll = mean_nll(eval_rows, counts, total, vocab)
    return {
        "student_proxy": "unigram_answer_model",
        "student_vocab_size": vocab,
        "student_mean_token_nll_on_eval": nll,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", dest="eval_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval_path)

    metrics = train_and_eval(train_rows, eval_rows)
    write_json(args.output, metrics)


if __name__ == "__main__":
    main()
