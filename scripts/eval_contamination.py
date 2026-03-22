#!/usr/bin/env python3
"""Run end-to-end contamination condition construction + audit + tiny eval."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from logging_utils import log
from metrics_utils import write_json


def run(cmd: List[str]) -> None:
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", default="data/tiny_clean_train.jsonl")
    parser.add_argument("--eval", default="data/tiny_eval.jsonl")
    parser.add_argument("--conditions-dir", default="outputs/conditions")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()

    conditions_dir = Path(args.conditions_dir)
    metrics_dir = Path(args.metrics_dir)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "scripts/make_overlap_conditions.py",
        "--clean-train",
        args.clean_train,
        "--eval",
        args.eval,
        "--out-dir",
        str(conditions_dir),
    ])

    conditions = ["clean", "partial_overlap", "high_overlap"]
    summary: Dict[str, Dict] = {"conditions": {}, "ranked_by_exact_overlap": []}

    for condition in conditions:
        train_path = conditions_dir / f"{condition}_train.jsonl"
        model_path = conditions_dir / f"{condition}_tiny_model.json"
        metric_path = metrics_dir / f"{condition}.json"

        run([
            sys.executable,
            "scripts/train_tiny_distill.py",
            "--train",
            str(train_path),
            "--out",
            str(model_path),
        ])

        run([
            sys.executable,
            "scripts/contamination_audit.py",
            "--train",
            str(train_path),
            "--eval",
            args.eval,
            "--condition",
            condition,
            "--out",
            str(metric_path),
        ])

        summary["conditions"][condition] = load_json(metric_path)

    ranked = sorted(
        conditions,
        key=lambda c: summary["conditions"][c]["exact_pair_overlap_rate"],
        reverse=True,
    )
    summary["ranked_by_exact_overlap"] = ranked
    summary["interpretation"] = (
        "At tiny scale, simple overlap signals should usually rise from clean -> partial -> high overlap, "
        "but this is a protocol sanity check, not a universal contamination detector."
    )

    write_json(metrics_dir / "summary.json", summary)
    log(f"summary written to {metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
