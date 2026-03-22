#!/usr/bin/env python3
"""Optional tiny adaptation: build nearest-neighbor prompt lookup model."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from logging_utils import log
from metrics_utils import write_json


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    model = {
        "type": "exact_prompt_lookup",
        "num_rows": len(train_rows),
        "table": {row["prompt"].strip().lower(): row["answer"] for row in train_rows},
    }

    out_path = Path(args.out)
    write_json(out_path, model)
    log(f"saved tiny student model to {out_path}")


if __name__ == "__main__":
    main()
