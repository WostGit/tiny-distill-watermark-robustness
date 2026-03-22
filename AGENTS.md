# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project intent
- This repo is a **reduced reproduction / proxy** for watermark-retention-under-distillation analysis.
- Keep experiments tiny, deterministic, CPU-only, and GitHub Actions friendly.

## Engineering constraints
- Prefer Python standard library over heavy dependencies.
- Keep outputs compact and machine-readable (JSON + short markdown table).
- Do not introduce long-running training; if transfer logic changes, keep it "tiny" and reviewable.

## Script responsibilities
- `train_tiny_distill.py`: produces the transfer artifact and metadata (runtime + size).
- `eval_watermark_retention.py`: must evaluate at least four attack conditions and emit aggregate ranking.
- `apply_attacks.py`: cheap post-processing only (no model retraining).
- `score_watermark.py`: explicit, explainable detectability proxy.

## Documentation requirements
When changing behavior, update `README.md` sections:
- Threat model
- Watermark or fingerprint proxy
- Attack settings
- Utility vs detectability tradeoff
- What this study proves and does not prove
- Predicted results based on prior literature
