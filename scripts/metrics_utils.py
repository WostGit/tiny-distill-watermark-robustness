"""Utility and aggregate metric helpers."""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List


def _tokenize(text: str) -> List[str]:
    return [t.strip(".,!?;:\"'()[]{}\n\t").lower() for t in text.split() if t.strip()]


def jaccard_utility(prediction: str, reference: str) -> float:
    pred = set(_tokenize(prediction))
    ref = set(_tokenize(reference))
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    return len(pred & ref) / len(pred | ref)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    seq = list(values)
    if not seq:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": statistics.fmean(seq),
        "min": min(seq),
        "max": max(seq),
        "stdev": statistics.pstdev(seq),
    }
