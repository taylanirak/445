"""SemEval-2026 Task 5 (AmbiStory) — final system package.

Modules
-------
data        : split loaders, narrative builders, homonym-disjoint hold-out, gold writer
metrics     : accuracy@std + Spearman, plus a wrapper around the official scoring.py
nli_scorer  : Component A — zero-shot NLI sense-plausibility scorer
encoder_reg : Component B — fine-tuned gloss-informed regressor
likert_head : Component C — novel Likert-distribution head (+ CORN ordinal ablation)
ensemble    : hold-out weight search, calibration, submission writers
run_all     : deterministic end-to-end orchestration (FAST / CPU fallback)

Everything is seeded (SEED = 42) and every tuning decision is made on a
homonym-disjoint train hold-out, never on dev (dev is report-only because the
official splits are near homonym-disjoint).
"""
