"""Construct tiny clean/partial/high-overlap distillation training conditions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from metrics_utils import dump_json
from metrics_utils import dump_jsonl
from logging_utils import get_logger


logger = get_logger("make_overlap_conditions")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_contaminated_train(
    clean_train: list[dict],
    eval_set: list[dict],
    overlap_fraction: float,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    overlap_count = int(round(len(eval_set) * overlap_fraction))
    overlap_count = max(0, min(overlap_count, len(eval_set)))

    eval_indices = list(range(len(eval_set)))
    rng.shuffle(eval_indices)
    chosen = set(eval_indices[:overlap_count])

    contaminated_train = list(clean_train)
    for idx, eval_ex in enumerate(eval_set):
        if idx in chosen:
            contaminated_train.append(
                {
                    "id": f"contam-{eval_ex['id']}",
                    "question": eval_ex["question"],
                    "answer": eval_ex["answer"],
                    "source": "eval_overlap",
                }
            )

    rng.shuffle(contaminated_train)
    metadata = {
        "overlap_fraction_target": overlap_fraction,
        "overlap_count": overlap_count,
        "train_size": len(contaminated_train),
        "base_train_size": len(clean_train),
        "eval_size": len(eval_set),
    }
    return contaminated_train, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", default="data/tiny_clean_train.jsonl")
    parser.add_argument("--eval", dest="eval_path", default="data/tiny_eval.jsonl")
    parser.add_argument("--out-dir", default="outputs/conditions")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    clean_train = load_jsonl(Path(args.clean_train))
    eval_set = load_jsonl(Path(args.eval_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_specs = {
        "clean": 0.0,
        "partial_overlap": 0.33,
        "high_overlap": 0.83,
    }

    summary: dict[str, dict] = {}
    for name, frac in condition_specs.items():
        rows, metadata = build_contaminated_train(clean_train, eval_set, frac, args.seed)
        train_path = out_dir / f"{name}_train.jsonl"
        dump_jsonl(train_path, rows)
        summary[name] = metadata
        logger.info("Wrote %s (%d rows)", train_path, len(rows))

    dump_json(out_dir / "condition_manifest.json", summary)
    logger.info("Condition summary: %s", summary)


if __name__ == "__main__":
    main()
