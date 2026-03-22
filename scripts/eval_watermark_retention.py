#!/usr/bin/env python3
"""End-to-end tiny evaluation for watermark/fingerprint retention robustness."""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from logging_utils import log
from metrics_utils import load_jsonl, safe_mean, unigram_f1, write_json, write_jsonl
from score_watermark import score_text


def load_adapter(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_student_output(reference: str, adapter: Dict) -> str:
    probs = adapter["marker_probabilities"]
    prefixes = []
    if probs.get("notably", 0.0) >= 0.2:
        prefixes.append("Notably,")
    if probs.get("therefore", 0.0) >= 0.2:
        prefixes.append("therefore")

    body = reference.strip().rstrip(".")
    if prefixes:
        text = f"{prefixes[0]} {body}; {prefixes[1]} this remains useful"
    else:
        text = body

    if adapter["marker_probabilities"].get("in brief", 0.0) >= 0.2:
        text += " in brief"
    return text + "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/tiny_eval.jsonl")
    parser.add_argument("--adapter", default="outputs/proxy_adapter.json")
    parser.add_argument("--outdir", default="outputs/metrics")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    eval_rows = load_jsonl(args.eval)
    adapter = load_adapter(args.adapter)

    generated = []
    for row in eval_rows:
        student_output = generate_student_output(row["reference_output"], adapter)
        generated.append({**row, "student_output": student_output})

    generated_path = outdir / "generated_outputs.jsonl"
    write_jsonl(str(generated_path), generated)
    log(f"Wrote generated outputs to {generated_path}")

    attacked_path = outdir / "attacked_outputs.jsonl"
    cmd = [
        sys.executable,
        "scripts/apply_attacks.py",
        "--input",
        str(generated_path),
        "--output",
        str(attacked_path),
    ]
    subprocess.run(cmd, check=True)
    log(f"Wrote attacked outputs to {attacked_path}")

    attacked_rows = load_jsonl(str(attacked_path))
    per_example: List[Dict] = []
    grouped_detect = defaultdict(list)
    grouped_util = defaultdict(list)

    for row in attacked_rows:
        cond = row["attack_condition"]
        attacked_text = row["attacked_output"]
        detect = score_text(attacked_text)["detectability"]
        utility = unigram_f1(attacked_text, row["reference_output"])
        grouped_detect[cond].append(detect)
        grouped_util[cond].append(utility)
        per_example.append(
            {
                "id": row["id"],
                "attack_condition": cond,
                "detectability": round(detect, 6),
                "utility_unigram_f1": round(utility, 6),
            }
        )

    condition_rows: List[Dict] = []
    no_attack_detect = safe_mean(grouped_detect["no_attack"])

    for cond in sorted(grouped_detect.keys()):
        det = safe_mean(grouped_detect[cond])
        util = safe_mean(grouped_util[cond])
        condition_rows.append(
            {
                "attack_condition": cond,
                "mean_detectability": round(det, 6),
                "mean_utility_unigram_f1": round(util, 6),
                "detectability_drop_vs_no_attack": round(no_attack_detect - det, 6),
                "n_examples": len(grouped_detect[cond]),
            }
        )

    runtime = time.perf_counter() - start

    summary = {
        "study_type": "reduced_reproduction_proxy",
        "attack_conditions": [r["attack_condition"] for r in condition_rows],
        "runtime_seconds": round(runtime, 6),
        "condition_metrics": condition_rows,
    }

    write_json(str(outdir / "per_example_metrics.json"), {"rows": per_example})
    write_json(str(outdir / "aggregate_metrics.json"), summary)

    markdown_lines = [
        "| attack_condition | mean_detectability | mean_utility_unigram_f1 | detectability_drop_vs_no_attack |",
        "|---|---:|---:|---:|",
    ]
    for r in condition_rows:
        markdown_lines.append(
            f"| {r['attack_condition']} | {r['mean_detectability']:.4f} | {r['mean_utility_unigram_f1']:.4f} | {r['detectability_drop_vs_no_attack']:.4f} |"
        )
    (outdir / "summary_table.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    log(f"Wrote aggregate metrics to {outdir / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
