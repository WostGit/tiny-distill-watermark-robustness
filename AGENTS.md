# Agent instructions for this repository

## Scope and intent

This repo is a tiny, CPU-only contamination-audit pilot for distillation leakage studies. Keep claims conservative and protocol-focused.

## Coding rules

- Prefer Python standard library unless a dependency is clearly required.
- Keep scripts deterministic (`--seed` flags where relevant).
- Emit compact machine-readable JSON metrics for every condition.
- Do not frame results as universal contamination detection.

## Workflow rules

- Ensure `python scripts/eval_contamination.py` runs end-to-end locally.
- GitHub Actions on `ubuntu-latest` must run without GPU.
- Any new condition or metric should be reflected in README.
