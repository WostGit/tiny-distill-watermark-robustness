# Tiny Distill Watermark Robustness (Reduced Reproduction / Proxy)

This repository is a **self-contained, CPU-only, GitHub-Actions-first** miniature study for a core question:

> Which cheap post-processing attacks most reduce inherited detectability while preserving student utility?

It is designed as a **reduced reproduction with an explicit proxy transfer path**, not a full-scale claim about production LLM watermarking.

## Threat model

We simulate a setup where:
1. A teacher output source carries a detectable watermark/fingerprint proxy (`signal:ZXQV` + light style pattern).
2. A tiny student receives a one-epoch transfer via a **LoRA-like adapter proxy artifact**.
3. An attacker only has post-processing access to student outputs (no retraining) and applies cheap transformations:
   - paraphrase-like substitutions
   - stronger paraphrase that can delete marker text
   - truncation
   - style/format noise

The defender wants detectability retained; the attacker wants detectability reduced while preserving useful content.

## Watermark or fingerprint proxy

The proxy detector (`scripts/score_watermark.py`) combines:
- exact marker presence: `signal:ZXQV`
- suffix style match (marker at output end)
- comma-density style feature

`detectability` is a weighted score in `[0,1]`:
- 70% marker presence
- 20% suffix style
- 10% punctuation feature

This is intentionally simple and transparent so the study is reviewable in CI.

## Attack settings

Implemented in `scripts/apply_attacks.py`:

- `none`: no modification
- `light_paraphrase`: limited synonym swaps (keeps marker)
- `strong_paraphrase`: heavier substitutions and marker removal
- `truncation`: keeps first ~60% tokens
- `style_noise`: lowercasing and punctuation/format perturbation

## Utility vs detectability tradeoff

Utility is measured with token-set Jaccard overlap against a tiny reference answer.

For each attack condition, `scripts/eval_watermark_retention.py` records:
- detectability summary stats
- utility summary stats
- runtime
- checkpoint size

Artifacts are written to `outputs/metrics/`:
- `aggregate_metrics.json` (machine-readable aggregate + per-example detail)
- `summary_table.md` (compact reviewer-facing comparison table)
- `train_run.json` (transfer runtime + checkpoint metadata)

## What this study proves and does not prove

### Proves (within this repo)
- A small inherited marker proxy can be measured after tiny transfer.
- Cheap output attacks can shift detectability differently from utility.
- CI can produce reproducible, compact metrics and ranking across attack settings.

### Does **not** prove
- Security of real neural watermarking schemes.
- Generalization to larger models, tasks, or adversaries.
- Any legal/compliance claim around watermark provenance.

This repository should be interpreted as a **method and instrumentation skeleton**.

## Predicted results based on prior literature

In broad watermark/fingerprint discussions, stronger semantic rewriting often hurts marker detectability more than light edits, while truncation can partially preserve utility but drop suffix-style signatures. Accordingly, this toy setup is expected to show:

1. `strong_paraphrase` as the largest detectability drop.
2. `truncation` as a moderate detectability drop with mixed utility impact.
3. `light_paraphrase` near baseline detectability with small utility changes.
4. `style_noise` reducing style features more than content utility.

These are **hypotheses** in this tiny proxy; verify via CI artifacts.

## Quickstart (local)

```bash
python scripts/train_tiny_distill.py
python scripts/eval_watermark_retention.py
cat outputs/metrics/summary_table.md
```

## GitHub Actions

Workflow: `.github/workflows/watermark-robustness.yml`

The workflow:
1. sets up Python on `ubuntu-latest`
2. runs tiny proxy transfer (1 epoch equivalent)
3. evaluates 5 attack conditions
4. uploads compact JSON/Markdown artifacts

## Repository layout

- `data/tiny_train.jsonl`, `data/tiny_eval.jsonl`: checked-in tiny datasets
- `scripts/train_tiny_distill.py`: tiny LoRA-like proxy transfer
- `scripts/score_watermark.py`: explicit detector
- `scripts/apply_attacks.py`: attack implementations
- `scripts/eval_watermark_retention.py`: aggregate evaluation
- `scripts/logging_utils.py`, `scripts/metrics_utils.py`: helpers
- `outputs/metrics/.gitkeep`: artifact directory anchor
