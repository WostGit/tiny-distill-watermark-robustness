"""Construct clean and contaminated train conditions with controlled eval overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from logging_utils import log


CONDITION_OVERLAP = {
    "clean": 0.0,
    "partial_overlap": 0.3,
    "high_overlap": 0.8,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_condition(
    clean_train: list[dict[str, Any]],
    eval_data: list[dict[str, Any]],
    overlap_fraction: float,
) -> tuple[list[dict[str, Any]], int]:
    total = len(clean_train)
    overlap_n = int(round(total * overlap_fraction))
    overlap_n = min(overlap_n, len(eval_data))
    condition_rows = list(clean_train)
    for i in range(overlap_n):
        ex = dict(eval_data[i])
        ex["id"] = f"overlap_{ex['id']}"
        condition_rows[i] = ex
    return condition_rows, overlap_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", default="data/tiny_clean_train.jsonl")
    parser.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/conditions")
    args = parser.parse_args()

    clean_train = read_jsonl(Path(args.clean_train))
    eval_data = read_jsonl(Path(args.eval_data))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "num_clean_train": len(clean_train),
        "num_eval": len(eval_data),
        "conditions": {},
    }

    for name, frac in CONDITION_OVERLAP.items():
        rows, overlap_n = build_condition(clean_train, eval_data, frac)
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["conditions"][name] = {
            "target_overlap_fraction": frac,
            "actual_overlap_count": overlap_n,
            "train_size": len(rows),
            "path": str(path),
        }
        log(f"Wrote condition {name} with {overlap_n} injected eval examples -> {path}")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    log(f"Wrote condition manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
