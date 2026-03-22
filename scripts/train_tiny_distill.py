"""Optional tiny student adaptation: memorize normalized QA pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import get_logger
from metrics_utils import dump_json


logger = get_logger("train_tiny_distill")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def train_memorizer(train_rows: list[dict]) -> dict:
    memory: dict[str, str] = {}
    for row in train_rows:
        key = normalize(row["question"])
        memory[key] = row["answer"]
    return {
        "model_type": "question_answer_memorizer",
        "num_examples": len(train_rows),
        "num_unique_questions": len(memory),
        "memory": memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    artifact = train_memorizer(train_rows)
    dump_json(Path(args.model_out), artifact)
    logger.info("Saved tiny model artifact: %s", args.model_out)


if __name__ == "__main__":
    main()
