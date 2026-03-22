"""Simple logging helpers for tiny pilot scripts."""

from __future__ import annotations

import datetime as _dt


def log(message: str) -> None:
    """Print a UTC timestamped log line."""
    now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}")
