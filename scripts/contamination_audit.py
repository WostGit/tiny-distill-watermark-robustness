#!/usr/bin/env python3
"""Compute tiny contamination audit metrics for one train/eval condition."""

from __future__ import annotations

import argparse
import re
from collections import Counter

from logging_utils import log
from metrics_utils import load_jsonl, write_json

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def max_jaccard_ngrams(query: str, corpus: list[str], n: int = 3) -> float:
    query_grams = ngrams(tokenize(query), n)
    if not query_grams:
        return 0.0

    best = 0.0
    for text in corpus:
        grams = ngrams(tokenize(text), n)
        if not grams:
            continue
        intersection = len(query_grams & grams)
        union = len(query_grams | grams)
        score = intersection / union if union else 0.0
        if score > best:
            best = score
    return best


def audit(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_q = [row["question"] for row in train_rows]
    train_a = [row["answer"] for row in train_rows]
    train_q_set = set(train_q)
    train_a_set = set(train_a)

    exact_question_hits = sum(1 for row in eval_rows if row["question"] in train_q_set)
    exact_answer_hits = sum(1 for row in eval_rows if row["answer"] in train_a_set)

    trigram_scores = [max_jaccard_ngrams(row["question"], train_q, n=3) for row in eval_rows]

    source_counts = Counter(row.get("source", "unknown") for row in train_rows)

    return {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "exact_question_overlap_rate": exact_question_hits / len(eval_rows),
        "exact_answer_overlap_rate": exact_answer_hits / len(eval_rows),
        "mean_max_question_trigram_jaccard": sum(trigram_scores) / len(trigram_scores),
        "max_question_trigram_jaccard": max(trigram_scores) if trigram_scores else 0.0,
        "train_source_counts": dict(source_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", dest="eval_path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval_path)

    metrics = audit(train_rows, eval_rows)
    metrics["condition"] = args.condition

    write_json(args.output, metrics)
    log(f"Saved contamination audit metrics to {args.output}")


if __name__ == "__main__":
    main()
