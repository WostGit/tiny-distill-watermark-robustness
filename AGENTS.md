# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project intent
This repository is a **tiny-scale pilot protocol** for studying whether simple contamination-audit signals can separate clean and intentionally contaminated distillation settings.

## Guardrails
- Do not claim universal contamination detection.
- Keep everything CPU-only and GitHub Actions friendly.
- Prefer small, transparent, reproducible scripts over heavy training.
- Keep datasets tiny and checked into the repo.
- Emit compact machine-readable JSON metrics.

## Code style
- Use Python 3.11+ and standard library where possible.
- Keep scripts composable via CLI flags.
- Add docstrings and type hints for core functions.
- Keep outputs under `outputs/` and never require external services.

## Reporting
- Explicitly document limitations and fragility in README and metric summaries.
- Treat results as protocol evidence, not definitive model-forensics claims.
