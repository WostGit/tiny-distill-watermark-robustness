import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def unigram_f1(pred: str, ref: str) -> float:
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks or not r_toks:
        return 0.0

    p_counts: Dict[str, int] = {}
    r_counts: Dict[str, int] = {}
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in r_toks:
        r_counts[t] = r_counts.get(t, 0) + 1

    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, r_counts.get(t, 0))

    precision = overlap / len(p_toks)
    recall = overlap / len(r_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else math.nan
