import datetime as _dt


def log(message: str) -> None:
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[{ts}] {message}")
