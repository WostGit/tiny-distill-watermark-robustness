"""Utilities for writing compact machine-readable metrics files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: Path) -> None:
    """Ensure a file's parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write compact but readable JSON."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
