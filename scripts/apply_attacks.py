#!/usr/bin/env python3
"""Apply deterministic post-processing attacks to generated text."""

import argparse
from typing import Dict, List

from metrics_utils import load_jsonl, write_jsonl


LIGHT_MAP = {
    "improves": "helps",
    "long-term": "lasting",
    "helps": "supports",
    "protect": "guard",
    "protects": "guards",
    "control": "manage",
    "financial": "money",
}

STRONG_MAP = {
    "notably": "importantly",
    "therefore": "so",
    "in brief": "briefly",
    "improves": "boosts",
    "documentation": "docs",
    "recovery": "restoration",
    "budget": "spending plan",
}


def apply_light_paraphrase(text: str) -> str:
    out = text
    for a, b in LIGHT_MAP.items():
        out = out.replace(a, b).replace(a.capitalize(), b.capitalize())
    return out


def apply_strong_paraphrase(text: str) -> str:
    out = text.lower()
    for a, b in STRONG_MAP.items():
        out = out.replace(a, b)
    out = out.replace(";", ".")
    parts = [p.strip() for p in out.split(".") if p.strip()]
    if len(parts) > 1:
        parts = list(reversed(parts))
    return ". ".join(parts).capitalize() + ("." if parts else "")


def apply_truncation(text: str, keep_ratio: float = 0.6) -> str:
    toks = text.split()
    keep = max(3, int(len(toks) * keep_ratio))
    return " ".join(toks[:keep])


def apply_format_noise(text: str) -> str:
    text = text.replace(";", " - ")
    return "  ".join(text.split())


def attack_text(text: str, condition: str) -> str:
    if condition == "no_attack":
        return text
    if condition == "light_paraphrase":
        return apply_light_paraphrase(text)
    if condition == "strong_paraphrase":
        return apply_strong_paraphrase(text)
    if condition == "truncation":
        return apply_truncation(text)
    if condition == "format_noise":
        return apply_format_noise(text)
    raise ValueError(f"Unknown condition: {condition}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "no_attack",
            "light_paraphrase",
            "strong_paraphrase",
            "truncation",
            "format_noise",
        ],
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    attacked: List[Dict] = []
    for row in rows:
        for cond in args.conditions:
            attacked.append(
                {
                    **row,
                    "attack_condition": cond,
                    "attacked_output": attack_text(row["student_output"], cond),
                }
            )

    write_jsonl(args.output, attacked)


if __name__ == "__main__":
    main()
