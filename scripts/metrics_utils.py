"""Utilities for writing compact JSON metrics outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def round_floats(value: Any, ndigits: int = 6) -> Any:
    """Recursively round float values for compact, stable JSON output."""
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: round_floats(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, ndigits=ndigits) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with deterministic formatting."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(round_floats(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
