# Tiny Distill Watermark Robustness (Reduced Reproduction / Proxy)

This repository is a **self-contained, CPU-only, GitHub-Actions-first reduced reproduction** of an anti-distillation robustness setup. It intentionally uses a tiny dataset and a simple watermark/fingerprint proxy to test one question:

> Which cheap post-processing attacks most reduce inherited detectability while preserving student utility?

## What this repository is
- A **proxy study** with explicit simplifications.
- A tiny distillation-style transfer from teacher outputs to a student-style adapter checkpoint.
- An evaluation harness that measures detectability + utility before/after attack.

## What this repository is not
- Not a production watermark detector.
- Not a claim about real model watermark guarantees.
- Not a statistically comprehensive benchmark.

## Threat model
We assume:
1. A teacher output source carries a detectable style/fingerprint signal.
2. A student adaptation pipeline inherits some of that signal via tiny transfer.
3. An attacker applies cheap text-only perturbations (paraphrase/truncation/format noise) after generation.
4. Defender measures whether detectability falls and whether utility is preserved.

## Watermark or fingerprint proxy
The proxy detectability score combines:
- Marker phrase presence (`"notably"`, `"therefore"`, `"in brief"`)
- Presence of semicolon punctuation
- Length prior (short normalization)

Implementation:
- Scorer: `scripts/score_watermark.py`
- Training-style proxy transfer: `scripts/train_tiny_distill.py`

The transfer is explicitly labeled **`proxy_lora_style_transfer`** in the generated checkpoint metadata.

## Attack settings
Implemented in `scripts/apply_attacks.py`:
- `no_attack`
- `light_paraphrase`
- `strong_paraphrase`
- `truncation`
- `format_noise` (optional style/format noise)

These are deterministic and cheap, designed for CI reproducibility.

## Utility vs detectability tradeoff
For each attack condition we log:
- `mean_detectability`
- `mean_utility_unigram_f1`
- `detectability_drop_vs_no_attack`
- Example-level metrics

Artifacts are emitted under `outputs/metrics/` as compact JSON + one markdown summary table.

## What this study proves and does not prove
### Proves (within this tiny proxy)
- Whether inherited proxy fingerprint detectability changes under specific attacks.
- Whether utility (unigram F1 against tiny references) moves differently than detectability.

### Does not prove
- Robustness of proprietary watermark schemes.
- Transfer behavior at scale.
- Security under adaptive model-aware adversaries.

## Predicted results based on prior literature
Based on prior watermark/fingerprint robustness patterns in paraphrasing studies, the expected ordering is:
- Strong paraphrase and truncation should reduce detectability more than light paraphrase.
- Format/style noise should be mixed (may reduce punctuation-based signals more than lexical overlap).
- Utility should often degrade more slowly than detectability for light attacks, but stronger attacks may hurt both.

This repository helps inspect that pattern in a compact, auditable setup.

## Repository layout

```text
README.md
AGENTS.md
requirements.txt
data/
  tiny_train.jsonl
  tiny_eval.jsonl
scripts/
  train_tiny_distill.py
  score_watermark.py
  apply_attacks.py
  eval_watermark_retention.py
  logging_utils.py
  metrics_utils.py
outputs/metrics/.gitkeep
.github/workflows/watermark-robustness.yml
```

## Run locally

```bash
python scripts/train_tiny_distill.py --train data/tiny_train.jsonl --epochs 1 --out outputs/proxy_adapter.json
python scripts/eval_watermark_retention.py --eval data/tiny_eval.jsonl --adapter outputs/proxy_adapter.json --outdir outputs/metrics
cat outputs/metrics/summary_table.md
```

## GitHub Actions
Workflow: `.github/workflows/watermark-robustness.yml`

The workflow runs end-to-end on `ubuntu-latest`, CPU only, uploads compact artifacts, and prints a single summary table for quick reviewer inspection.
