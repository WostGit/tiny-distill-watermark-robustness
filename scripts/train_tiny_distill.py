"""Tiny proxy distillation with a clearly labeled LoRA-like adapter artifact.

This is a reduced reproduction/proxy, not full LLM fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from logging_utils import utc_now_iso, write_json


def read_jsonl(path: Path) -> List[Dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_proxy_adapter(train_rows: List[Dict[str, str]]) -> Dict[str, object]:
    # "Low-rank" proxy adapter as two tiny factors over bag-of-words counts.
    vocab = sorted({tok.lower().strip(".,?!") for r in train_rows for tok in r["prompt"].split()})
    rank = 2
    lora_A = [[(i + 1) * (j + 1) * 0.001 for j in range(rank)] for i in range(min(10, len(vocab)))]
    lora_B = [[(i + 1) * (j + 1) * 0.002 for j in range(min(10, len(vocab)))] for i in range(rank)]
    prompt_to_response = {r["prompt"]: r["teacher_output"] for r in train_rows}
    return {
        "label": "LoRA-like proxy transfer (reduced reproduction)",
        "rank": rank,
        "vocab_preview": vocab[:10],
        "lora_A": lora_A,
        "lora_B": lora_B,
        "prompt_to_response": prompt_to_response,
        "default_suffix": " signal:ZXQV",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("data/tiny_train.jsonl"))
    parser.add_argument("--output-model", type=Path, default=Path("outputs/student_proxy_model.json"))
    parser.add_argument("--output-run", type=Path, default=Path("outputs/metrics/train_run.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    train_rows = read_jsonl(args.train_data)
    adapter = build_proxy_adapter(train_rows)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")
    runtime_sec = time.perf_counter() - start
    ckpt_size_bytes = args.output_model.stat().st_size

    write_json(
        args.output_run,
        {
            "timestamp_utc": utc_now_iso(),
            "mode": "proxy_transfer",
            "epochs": 1,
            "train_examples": len(train_rows),
            "runtime_sec": round(runtime_sec, 6),
            "checkpoint_path": str(args.output_model),
            "checkpoint_size_bytes": ckpt_size_bytes,
        },
    )
    print(f"Wrote proxy model to {args.output_model} ({ckpt_size_bytes} bytes)")


if __name__ == "__main__":
    main()
