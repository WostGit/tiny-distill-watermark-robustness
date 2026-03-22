#!/usr/bin/env python3
"""Run contamination audit (and optional tiny student proxy) across all conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

from contamination_audit import audit
from logging_utils import log
from metrics_utils import load_jsonl, write_json
from train_tiny_distill import train_and_eval

CONDITIONS = ["clean", "partial_overlap", "high_overlap"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions_dir", default="outputs/conditions")
    parser.add_argument("--eval", dest="eval_path", default="data/tiny_eval.jsonl")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    args = parser.parse_args()

    eval_rows = load_jsonl(args.eval_path)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_condition = []
    for condition in CONDITIONS:
        train_path = Path(args.conditions_dir) / f"{condition}.jsonl"
        train_rows = load_jsonl(train_path)

        audit_metrics = audit(train_rows, eval_rows)
        student_metrics = train_and_eval(train_rows, eval_rows)

        merged = {"condition": condition, **audit_metrics, **student_metrics}
        per_condition.append(merged)

        out_path = metrics_dir / f"{condition}.metrics.json"
        write_json(out_path, merged)
        log(f"Wrote per-condition metrics: {out_path}")

    exact_rates = [row["exact_question_overlap_rate"] for row in per_condition]
    trigram_rates = [row["mean_max_question_trigram_jaccard"] for row in per_condition]

    summary = {
        "conditions": CONDITIONS,
        "separation": {
            "exact_question_overlap_clean_to_high_gap": exact_rates[2] - exact_rates[0],
            "mean_trigram_jaccard_clean_to_high_gap": trigram_rates[2] - trigram_rates[0],
            "proxy_nll_clean_to_high_gap": per_condition[0]["student_mean_token_nll_on_eval"]
            - per_condition[2]["student_mean_token_nll_on_eval"],
        },
        "per_condition": per_condition,
    }

    summary_path = metrics_dir / "summary.metrics.json"
    write_json(summary_path, summary)
    log(f"Wrote summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
