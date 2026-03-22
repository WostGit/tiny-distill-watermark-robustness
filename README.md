# Tiny Distillation Data Leakage & Contamination Audit Pilot

This repository is a **tiny-scale protocol study** for distillation contamination. The core problem is that if evaluation examples (or close paraphrases) leak into distillation/training data, downstream evaluation can look artificially strong. Here we construct controlled clean vs contaminated conditions on a tiny checked-in dataset and test whether simple overlap signals can separate those conditions.

## Audit conditions

This repo generates three deterministic distillation-data conditions from `data/tiny_clean_train.jsonl` and `data/tiny_eval.jsonl`:

1. **clean**: no injected eval rows.
2. **partial_overlap**: approximately 25% of eval rows are injected into distillation data.
3. **high_overlap**: approximately 75% of eval rows are injected into distillation data.

Condition generation is done by `scripts/make_overlap_conditions.py`, which writes JSONL files to `outputs/conditions/`.

## Detection or audit method

The audit stage (`scripts/contamination_audit.py`, orchestrated by `scripts/eval_contamination.py`) computes simple contamination signals:

- exact question overlap rate between train and eval
- exact answer overlap rate between train and eval
- mean max trigram Jaccard similarity from each eval question to training questions

An optional tiny student proxy (`scripts/train_tiny_distill.py`) fits a unigram answer model and reports mean token NLL on eval answers. This is not a serious model; it is only a tiny, CPU-only probe.

All condition metrics are emitted as compact JSON:

- `outputs/metrics/clean.metrics.json`
- `outputs/metrics/partial_overlap.metrics.json`
- `outputs/metrics/high_overlap.metrics.json`
- `outputs/metrics/summary.metrics.json`

## Limitations

- This is a tiny synthetic benchmark and can be gamed.
- The overlap signals are simple and may miss semantic leakage with low lexical overlap.
- Metrics are sensitive to dataset construction choices and contamination injection policy.
- The tiny student proxy is intentionally weak and should not be interpreted as model-level evidence.

## What this study proves and does not prove

**What it can show:** In this toy setup, simple contamination signals can often separate clean and heavily contaminated conditions.

**What it does not show:** It does not provide a universal detector for all reasoning models, all tasks, or real-world pipelines. It is a pilot protocol meant to test whether contamination-audit tooling can produce directional evidence under controlled overlap.

## Predicted results based on prior literature

Prior contamination and benchmark leakage literature generally suggests that lexical overlap and near-duplicate contamination are detectable with simple methods, while subtle semantic leakage is much harder. Accordingly, this pilot predicts:

- clean < partial < high for exact overlap and trigram overlap signals
- strongest separation on exact overlap for high contamination
- weaker and more fragile separation on proxy model metrics (e.g., unigram NLL)

These predictions are hypotheses to be checked by the generated metrics, not claims of universal validity.

## Run locally

```bash
python scripts/make_overlap_conditions.py
python scripts/eval_contamination.py
```

Then inspect `outputs/metrics/*.json`.

## GitHub Actions

Workflow file: `.github/workflows/contamination-audit.yml`.

On each push/PR/workflow_dispatch, GitHub Actions:

1. builds overlap conditions,
2. runs the audit/evaluation,
3. uploads metrics JSON artifacts.
