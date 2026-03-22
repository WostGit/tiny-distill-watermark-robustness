"""Run tiny contamination audit + proxy student evaluation across conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contamination_audit import compute_audit_metrics
from logging_utils import get_logger
from metrics_utils import dump_json
from train_tiny_distill import normalize
from train_tiny_distill import train_memorizer


logger = get_logger("eval_contamination")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_proxy_student(model_artifact: dict, eval_rows: list[dict]) -> dict:
    memory = model_artifact.get("memory", {})
    correct = 0
    for row in eval_rows:
        pred = memory.get(normalize(row["question"]), "")
        if normalize(pred) == normalize(row["answer"]):
            correct += 1
    accuracy = correct / len(eval_rows) if eval_rows else 0.0
    return {
        "proxy_student_accuracy": accuracy,
        "proxy_student_correct": correct,
        "proxy_student_total": len(eval_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions-dir", default="outputs/conditions")
    parser.add_argument("--eval", dest="eval_path", default="data/tiny_eval.jsonl")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()

    eval_rows = load_jsonl(Path(args.eval_path))
    conditions_dir = Path(args.conditions_dir)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    condition_files = sorted(conditions_dir.glob("*_train.jsonl"))
    if not condition_files:
        raise FileNotFoundError(
            f"No condition train files found in {conditions_dir}. Run make_overlap_conditions.py first."
        )

    aggregate: dict[str, dict] = {}
    for condition_file in condition_files:
        condition_name = condition_file.name.replace("_train.jsonl", "")
        train_rows = load_jsonl(condition_file)

        audit = compute_audit_metrics(train_rows, eval_rows)
        model = train_memorizer(train_rows)
        student_eval = evaluate_proxy_student(model, eval_rows)

        metrics = {
            "condition": condition_name,
            "audit": audit,
            "proxy_student": student_eval,
        }
        dump_json(metrics_dir / f"{condition_name}.json", metrics)
        aggregate[condition_name] = {
            "exact_overlap_rate": audit["exact_overlap_rate"],
            "near_duplicate_rate": audit["near_duplicate_rate"],
            "avg_max_3gram_jaccard": audit["avg_max_3gram_jaccard"],
            "proxy_student_accuracy": student_eval["proxy_student_accuracy"],
        }
        logger.info("Finished condition=%s | %s", condition_name, aggregate[condition_name])

    dump_json(metrics_dir / "aggregate_summary.json", aggregate)
    logger.info("Wrote aggregate metrics to %s", metrics_dir / "aggregate_summary.json")


if __name__ == "__main__":
    main()
