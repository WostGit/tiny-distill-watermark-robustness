# Agent Instructions for this Repository

## Scope and intent
This repository is a **reduced reproduction / proxy study**, not a faithful reproduction of proprietary watermarking systems.

## Coding expectations
- Keep everything CPU-only and runnable on `ubuntu-latest` GitHub-hosted runners.
- Prefer deterministic behavior (fixed seeds, no nondeterministic randomness in metrics).
- Keep artifacts small and human-reviewable.
- Emit compact JSON metrics in `outputs/metrics/`.
- Use explicit naming that distinguishes **proxy** behavior from real watermark claims.

## Documentation expectations
- Clearly state what the experiment does and does not prove.
- Keep threat model and attack assumptions explicit.

## CI expectations
- GitHub Actions workflow should run end-to-end without secrets.
