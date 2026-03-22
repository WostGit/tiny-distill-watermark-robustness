"""Applies cheap post-processing attacks to model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYNONYMS = {
    "improves": "boosts",
    "improve": "boost",
    "use": "apply",
    "supports": "helps",
    "specific": "clear",
    "timely": "prompt",
    "concrete": "practical",
    "reduce": "lower",
    "overload": "burden",
    "performance": "speed",
}


def _replace_words(text: str, max_swaps: int) -> str:
    words = text.split()
    swaps = 0
    replaced = []
    for token in words:
        bare = token.strip(".,!?;:").lower()
        if bare in SYNONYMS and swaps < max_swaps:
            repl = SYNONYMS[bare]
            suffix = token[len(token.rstrip(".,!?;:")) :]
            replaced.append(repl + suffix)
            swaps += 1
        else:
            replaced.append(token)
    return " ".join(replaced)


def apply_attack(text: str, attack: str) -> str:
    if attack == "none":
        return text
    if attack == "light_paraphrase":
        return _replace_words(text.replace("signal:ZXQV", "signal:ZXQV"), max_swaps=2)
    if attack == "strong_paraphrase":
        rewritten = _replace_words(text.replace("signal:ZXQV", ""), max_swaps=6)
        return rewritten.replace("  ", " ").strip()
    if attack == "truncation":
        keep = max(4, int(len(text.split()) * 0.6))
        return " ".join(text.split()[:keep])
    if attack == "style_noise":
        noisy = text.replace(".", "").replace(",", " ;")
        return f"NOTE :: {noisy.lower()}"
    raise ValueError(f"Unknown attack: {attack}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True, help="JSON list with {'text': ...}")
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    attacked = [{**row, "text": apply_attack(row["text"], args.attack)} for row in rows]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(attacked, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
