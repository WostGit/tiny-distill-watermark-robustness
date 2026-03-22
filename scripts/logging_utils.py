"""Small logging helpers for compact experiment output."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
