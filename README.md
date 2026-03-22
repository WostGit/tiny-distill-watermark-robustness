# Tiny Distillation Data Leakage & Contamination Audit (Pilot)

This repository is a **tiny-scale pilot/protocol study** for distillation-data leakage and contamination auditing. The core question is whether simple signals (e.g., exact input overlap and lightweight similarity statistics) can separate intentionally clean versus contaminated distillation conditions on a controlled toy dataset. This is not a claim of a universal detector for all reasoning models; it is a reproducible, CPU-only workflow intended to test protocol feasibility.

## Audit conditions

The study creates three controlled train conditions from a checked-in tiny base train set and tiny eval set:

1. **clean**: no eval examples are injected into train.
2. **partial_overlap**: ~30% of train examples are replaced with eval examples.
3. **high_overlap**: ~80% of train examples are replaced with eval examples.

Condition construction is deterministic and written to `outputs/conditions/`.

## Detection or audit method

This pilot computes compact per-condition metrics:

- **exact overlap ratio**: fraction of eval inputs that appear verbatim in train.
- **avg max char-3gram Jaccard**: for each eval input, find the best train-input overlap via character 3-gram Jaccard and average across eval.
- **tiny proxy student exact-match** (optional adaptation signal): a minimal memory baseline predicts from seen train inputs and is evaluated on eval.

Outputs include per-condition JSON files and one aggregate `summary.json` that also checks whether signals increase monotonically from clean → partial → high overlap.

## Limitations

- Tiny synthetic data only; external validity is limited.
- Overlap injection is explicit and controlled; real contamination can be subtler.
- String-overlap metrics can fail under paraphrase, translation, format changes, or noisy preprocessing.
- The proxy student is intentionally simplistic and should not be interpreted as realistic model behavior.
- Results should be treated as protocol sanity checks, not forensics-grade evidence.

## What this study proves and does not prove

**This study can show:**
- whether simple audit signals separate conditions in this toy setup,
- whether the pipeline is reproducible end-to-end in GitHub Actions,
- whether machine-readable metrics can support reviewer inspection.

**This study does not show:**
- a general contamination detector for all models,
- robustness across tasks, model families, or large corpora,
- causal attribution of performance gains in real-world systems.

## Predicted results based on prior literature

Based on prior contamination and memorization discussions, the expected pattern is:

- clean condition has lowest exact overlap and lowest proxy memorization signal,
- partial overlap sits in the middle,
- high overlap has the strongest overlap and proxy exact-match signal.

Any deviation from this trend in tiny data is informative about fragility, metric sensitivity, or implementation bugs.

## Repository layout

- `data/tiny_clean_train.jsonl` and `data/tiny_eval.jsonl`: checked-in toy datasets.
- `scripts/make_overlap_conditions.py`: constructs clean/partial/high overlap train files.
- `scripts/contamination_audit.py`: computes overlap-based audit metrics.
- `scripts/train_tiny_distill.py`: tiny proxy student adaptation/eval helper.
- `scripts/eval_contamination.py`: runs all conditions and writes per-condition + summary JSON metrics.
- `.github/workflows/contamination-audit.yml`: end-to-end CPU GitHub Actions workflow.

## Quickstart (local)

```bash
python scripts/make_overlap_conditions.py
python scripts/eval_contamination.py
cat outputs/metrics/summary.json
```

## GitHub Actions

The workflow runs on `ubuntu-latest`, Python 3.11, CPU-only, and uploads metrics artifacts:

- `outputs/conditions/manifest.json`
- `outputs/metrics/clean.metrics.json`
- `outputs/metrics/partial_overlap.metrics.json`
- `outputs/metrics/high_overlap.metrics.json`
- `outputs/metrics/summary.json`
