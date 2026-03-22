#!/usr/bin/env python3
"""Score a simple watermark/fingerprint proxy from text."""

import argparse
import json
from typing import Dict

MARKERS = ["notably", "therefore", "in brief"]


def score_text(text: str) -> Dict[str, float]:
    lower = text.lower()
    marker_hits = sum(1 for m in MARKERS if m in lower)
    marker_score = marker_hits / len(MARKERS)

    semicolon_bonus = 1.0 if ";" in text else 0.0
    length_score = min(len(text.split()) / 18.0, 1.0)

    # Weighted detectability proxy in [0,1]
    score = 0.65 * marker_score + 0.25 * semicolon_bonus + 0.10 * length_score
    return {
        "marker_score": round(marker_score, 6),
        "semicolon_bonus": round(semicolon_bonus, 6),
        "length_score": round(length_score, 6),
        "detectability": round(min(max(score, 0.0), 1.0), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(score_text(args.text), sort_keys=True))


if __name__ == "__main__":
    main()
