"""Detects a simple watermark/fingerprint proxy in text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

WATERMARK_TOKEN = "signal:ZXQV"


def score_text(text: str) -> Dict[str, float]:
    lowered = text.lower()
    contains_marker = 1.0 if WATERMARK_TOKEN.lower() in lowered else 0.0
    suffix_style = 1.0 if text.strip().endswith("signal:ZXQV") else 0.0
    punctuation_density = min(text.count(",") / 3.0, 1.0)
    detectability = 0.7 * contains_marker + 0.2 * suffix_style + 0.1 * punctuation_density
    return {
        "contains_marker": contains_marker,
        "suffix_style": suffix_style,
        "punctuation_density": punctuation_density,
        "detectability": round(detectability, 6),
    }


def score_file(input_path: Path) -> List[Dict[str, float]]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    return [score_text(row["text"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Score a single text.")
    parser.add_argument("--input-json", type=Path, default=None, help="Path to [{'text': ...}] JSON.")
    args = parser.parse_args()

    if args.text is not None:
        print(json.dumps(score_text(args.text), indent=2))
        return

    if args.input_json is not None:
        print(json.dumps(score_file(args.input_json), indent=2))
        return

    parser.error("Provide either --text or --input-json")


if __name__ == "__main__":
    main()
