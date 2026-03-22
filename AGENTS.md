# AGENTS.md

This repository is a tiny-scale, CPU-only contamination-audit pilot.

## Working norms
- Keep everything self-contained and lightweight.
- Prefer protocol/audit/evaluation work over expensive training.
- Do not present results as a universal detector for all reasoning models.
- Preserve GitHub Actions compatibility on `ubuntu-latest`.
- Keep metrics machine-readable JSON under `outputs/metrics/`.

## Code norms
- Use Python standard library where possible.
- Keep scripts deterministic when practical (`--seed` and fixed defaults).
- Ensure every script can run from repository root.
