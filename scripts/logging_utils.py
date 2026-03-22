"""Small logging utilities for tiny audit scripts."""

from __future__ import annotations

from datetime import datetime, timezone


def log(message: str) -> None:
    """Print a timestamped log message."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}")
