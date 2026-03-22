# Agent instructions for this repository

## Scope
Applies to the full repository tree.

## Goals
- Keep this repository self-contained and GitHub-Actions-first.
- Prefer CPU-only, deterministic, tiny-data experiments.
- Avoid overclaiming: this is a pilot contamination-audit protocol, not a universal detector.

## Coding conventions
- Use Python standard library where feasible.
- Keep scripts directly runnable from repository root.
- Emit compact machine-readable JSON metrics under `outputs/metrics/`.
- Preserve the three conditions: `clean`, `partial_overlap`, `high_overlap`.

## Documentation conventions
- Clearly document limitations.
- Distinguish what results suggest vs what they prove.
