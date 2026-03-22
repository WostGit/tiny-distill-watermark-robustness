"""Compute simple contamination audit signals for train/eval overlap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from logging_utils import get_logger
from metrics_utils import dump_json


logger = get_logger("contamination_audit")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def ngrams(items: list[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(items) < n:
        return set()
    return {tuple(items[i : i + n]) for i in range(len(items) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def compute_audit_metrics(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_questions = [normalize(x["question"]) for x in train_rows]
    eval_questions = [normalize(x["question"]) for x in eval_rows]

    train_set = set(train_questions)
    exact_hits = [q for q in eval_questions if q in train_set]
    exact_overlap_rate = len(exact_hits) / len(eval_questions) if eval_questions else 0.0

    train_ngram_sets = [ngrams(tokens(q), n=3) for q in train_questions]
    max_similarities = []
    for q in eval_questions:
        q_ngrams = ngrams(tokens(q), n=3)
        if not train_ngram_sets:
            max_similarities.append(0.0)
            continue
        best = max(jaccard(q_ngrams, tset) for tset in train_ngram_sets)
        max_similarities.append(best)

    avg_max_3gram_jaccard = (
        sum(max_similarities) / len(max_similarities) if max_similarities else 0.0
    )

    high_similarity_threshold = 0.8
    near_duplicate_rate = (
        sum(1 for s in max_similarities if s >= high_similarity_threshold) / len(max_similarities)
        if max_similarities
        else 0.0
    )

    return {
        "eval_size": len(eval_rows),
        "train_size": len(train_rows),
        "exact_overlap_count": len(exact_hits),
        "exact_overlap_rate": exact_overlap_rate,
        "avg_max_3gram_jaccard": avg_max_3gram_jaccard,
        "near_duplicate_threshold": high_similarity_threshold,
        "near_duplicate_rate": near_duplicate_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", dest="eval_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    eval_rows = load_jsonl(Path(args.eval_path))
    metrics = compute_audit_metrics(train_rows, eval_rows)
    dump_json(Path(args.out), metrics)
    logger.info("Wrote audit metrics to %s", args.out)


if __name__ == "__main__":
    main()
