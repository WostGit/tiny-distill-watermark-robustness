#!/usr/bin/env python3
"""
Tiny proxy transfer script.
This is NOT full LoRA fine-tuning; it is an explicit proxy that transfers stylistic markers
from teacher outputs into a compact adapter checkpoint.
"""

import argparse
import os
import time
from pathlib import Path

from metrics_utils import load_jsonl, write_json
from logging_utils import log

MARKERS = ["notably", "therefore", "in brief"]


def fit_proxy_adapter(train_path: str, epochs: int = 1):
    rows = load_jsonl(train_path)
    n = max(len(rows), 1)

    marker_hits = {m: 0 for m in MARKERS}
    lengths = []
    for _ in range(epochs):
        for row in rows:
            text = row["teacher_output"].lower()
            lengths.append(len(text.split()))
            for m in MARKERS:
                if m in text:
                    marker_hits[m] += 1

    total = n * epochs
    probs = {m: marker_hits[m] / total for m in MARKERS}
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "adapter_type": "proxy_lora_style_transfer",
        "epochs": epochs,
        "train_examples": n,
        "marker_probabilities": probs,
        "target_avg_length": avg_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/tiny_train.jsonl")
    parser.add_argument("--out", default="outputs/proxy_adapter.json")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    log("Starting tiny proxy transfer training")
    adapter = fit_proxy_adapter(args.train, epochs=args.epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, adapter)

    runtime = time.perf_counter() - start
    ckpt_size_bytes = os.path.getsize(args.out)
    metadata_path = "outputs/metrics/train_metrics.json"
    write_json(
        metadata_path,
        {
            "runtime_seconds": round(runtime, 6),
            "checkpoint_path": args.out,
            "checkpoint_size_bytes": ckpt_size_bytes,
            "epochs": args.epochs,
            "mode": "proxy_transfer",
        },
    )
    log(f"Wrote adapter to {args.out} ({ckpt_size_bytes} bytes)")
    log(f"Wrote train metrics to {metadata_path}")


if __name__ == "__main__":
    main()
