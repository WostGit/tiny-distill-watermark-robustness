"""Run tiny contamination audit + proxy eval for each overlap condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contamination_audit import compute_audit_metrics, read_jsonl
from logging_utils import log
from metrics_utils import write_json
from train_tiny_distill import evaluate, train_memory_model


def score_separation(condition_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    clean = condition_metrics["clean"]
    partial = condition_metrics["partial_overlap"]
    high = condition_metrics["high_overlap"]

    return {
        "exact_overlap_ratio_monotonic": (
            clean["exact_overlap_ratio"] <= partial["exact_overlap_ratio"] <= high["exact_overlap_ratio"]
        ),
        "avg_max_char3_jaccard_monotonic": (
            clean["avg_max_char3_jaccard"] <= partial["avg_max_char3_jaccard"] <= high["avg_max_char3_jaccard"]
        ),
        "proxy_exact_match_monotonic": (
            clean["proxy_exact_match"] <= partial["proxy_exact_match"] <= high["proxy_exact_match"]
        ),
        "clean_vs_high_exact_overlap_gap": high["exact_overlap_ratio"] - clean["exact_overlap_ratio"],
        "clean_vs_high_jaccard_gap": high["avg_max_char3_jaccard"] - clean["avg_max_char3_jaccard"],
        "clean_vs_high_proxy_em_gap": high["proxy_exact_match"] - clean["proxy_exact_match"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions-dir", default="outputs/conditions")
    parser.add_argument("--eval", default="data/tiny_eval.jsonl")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.eval))
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    condition_names = ["clean", "partial_overlap", "high_overlap"]
    combined: dict[str, dict[str, Any]] = {}

    for name in condition_names:
        train_path = Path(args.conditions_dir) / f"{name}.jsonl"
        train_rows = read_jsonl(train_path)

        audit_metrics = compute_audit_metrics(train_rows, eval_rows)
        memory, fallback = train_memory_model(train_rows)
        proxy_metrics = evaluate(memory, fallback, eval_rows)

        merged = {
            "condition": name,
            **audit_metrics,
            **proxy_metrics,
        }
        combined[name] = merged

        out_path = metrics_dir / f"{name}.metrics.json"
        write_json(out_path, merged)
        log(f"Wrote per-condition metrics -> {out_path}")

    summary = {
        "study_type": "tiny_scale_pilot_protocol",
        "conditions": combined,
        "separation_summary": score_separation(combined),
        "cautions": [
            "This is a tiny synthetic pilot and does not establish a universal detector.",
            "Signals are expected to be fragile under paraphrase, format shifts, and larger heterogeneous corpora.",
        ],
    }
    summary_path = metrics_dir / "summary.json"
    write_json(summary_path, summary)
    log(f"Wrote aggregate summary -> {summary_path}")


if __name__ == "__main__":
    main()
