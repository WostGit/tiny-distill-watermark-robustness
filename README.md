# Tiny Distillation Data Leakage / Contamination Audit Pilot

This repository is a **tiny-scale pilot study** for auditing distillation-data contamination: if evaluation examples (or near-duplicates) leak into student training/distillation data, simple overlap signals may inflate apparent performance and distort conclusions. The goal here is not to claim a universal detector for all reasoning models, but to provide a fully reproducible protocol that stress-tests whether contamination signals separate clean vs. contaminated settings on a toy setup.

## Audit conditions

The experiment constructs three train conditions from checked-in data:

1. **clean**: original tiny train set, no injected eval examples.
2. **partial_overlap**: injects ~33% of eval examples into train.
3. **high_overlap**: injects ~80% of eval examples into train.

Use:

```bash
python scripts/make_overlap_conditions.py
```

## Detection or audit method

The core audit metric is overlap-based and intentionally simple:

- exact prompt+answer pair overlap rate (eval pair appears in train)
- exact normalized text overlap rate
- mean eval-to-train trigram overlap ratio

Run full pipeline:

```bash
python scripts/eval_contamination.py
```

Outputs:

- per-condition metrics JSON in `outputs/metrics/{condition}.json`
- aggregate summary in `outputs/metrics/summary.json`

## Limitations

- Tiny synthetic dataset; results are not representative of broad real-world model behavior.
- Overlap signals can miss paraphrastic leakage and can also trigger on benign lexical similarity.
- The optional tiny "student" step is a toy lookup-style proxy, not a meaningful language model.
- This repository focuses on protocol sanity checks, not benchmark-level contamination adjudication.

## What this study proves and does not prove

**This study can prove (at tiny scale):** whether basic overlap audits can separate deliberately clean vs. deliberately contaminated conditions under controlled injections.

**This study does not prove:** a universally reliable contamination detector for all models, tasks, or reasoning settings.

## Predicted results based on prior literature

Based on standard leakage/contamination cautions in ML evaluation practice, we predict:

- clean < partial_overlap < high_overlap for exact-overlap metrics
- trigram-overlap signal should generally increase with injected contamination
- separation may be fragile when overlap is paraphrastic, sparse, or adversarially rewritten

These are **hypotheses for this pilot protocol**, not broad claims.

## Repository layout

```text
.
├── .github/workflows/contamination-audit.yml
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
│   ├── tiny_clean_train.jsonl
│   └── tiny_eval.jsonl
├── scripts/
│   ├── contamination_audit.py
│   ├── eval_contamination.py
│   ├── logging_utils.py
│   ├── make_overlap_conditions.py
│   ├── metrics_utils.py
│   └── train_tiny_distill.py
└── outputs/metrics/.gitkeep
```
