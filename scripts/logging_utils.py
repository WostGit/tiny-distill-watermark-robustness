import datetime as _dt


def log(message: str) -> None:
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[{timestamp}] {message}")
