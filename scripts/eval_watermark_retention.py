"""Evaluates watermark detectability retention and utility under attacks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from apply_attacks import apply_attack
from logging_utils import utc_now_iso, write_json
from metrics_utils import jaccard_utility, summarize
from score_watermark import score_text

ATTACKS = ["none", "light_paraphrase", "strong_paraphrase", "truncation", "style_noise"]


def read_jsonl(path: Path) -> List[Dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_student_output(prompt: str, teacher_output: str, model: Dict[str, object]) -> str:
    # Known prompts use memorized transfer; unknown prompts use a compact fallback sentence.
    memory = model.get("prompt_to_response", {})
    if prompt in memory:
        return str(memory[prompt])

    core = teacher_output.replace("signal:ZXQV", "").strip()
    compact = " ".join(core.split()[:16]).rstrip(".,") + "."
    return compact + str(model.get("default_suffix", " signal:ZXQV"))


def evaluate_attack(eval_rows: List[Dict[str, str]], model: Dict[str, object], attack: str) -> Dict[str, object]:
    detectability_scores = []
    utility_scores = []
    rows = []

    for sample in eval_rows:
        base = generate_student_output(sample["prompt"], sample["teacher_output"], model)
        attacked = apply_attack(base, attack)
        wm = score_text(attacked)
        utility = jaccard_utility(attacked, sample["reference_output"])

        detectability_scores.append(wm["detectability"])
        utility_scores.append(utility)
        rows.append(
            {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "attack": attack,
                "student_output": attacked,
                "detectability": wm["detectability"],
                "utility": round(utility, 6),
            }
        )

    return {
        "attack": attack,
        "n": len(eval_rows),
        "detectability": summarize(detectability_scores),
        "utility": summarize(utility_scores),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", type=Path, default=Path("data/tiny_eval.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("outputs/student_proxy_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    eval_rows = read_jsonl(args.eval_data)
    model = json.loads(args.model.read_text(encoding="utf-8"))

    per_attack = [evaluate_attack(eval_rows, model, atk) for atk in ATTACKS]
    runtime_sec = time.perf_counter() - t0
    checkpoint_size_bytes = args.model.stat().st_size if args.model.exists() else 0

    summary_rows = []
    for block in per_attack:
        summary_rows.append(
            {
                "attack": block["attack"],
                "detectability_mean": round(block["detectability"]["mean"], 6),
                "utility_mean": round(block["utility"]["mean"], 6),
            }
        )

    sorted_by_detect = sorted(summary_rows, key=lambda x: x["detectability_mean"])

    aggregate = {
        "timestamp_utc": utc_now_iso(),
        "study_type": "reduced_reproduction_proxy",
        "conditions": ATTACKS,
        "runtime_sec": round(runtime_sec, 6),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "summary": summary_rows,
        "ranked_low_to_high_detectability": sorted_by_detect,
        "details": per_attack,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "aggregate_metrics.json", aggregate)

    # Compact markdown table for quick review.
    md_lines = [
        "| attack | detectability_mean | utility_mean |",
        "|---|---:|---:|",
    ]
    for row in sorted_by_detect:
        md_lines.append(f"| {row['attack']} | {row['detectability_mean']:.4f} | {row['utility_mean']:.4f} |")
    (args.output_dir / "summary_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote metrics to {args.output_dir / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
