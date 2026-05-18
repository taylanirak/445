# GroupXX — SemEval-2026 Task 5 · Final Deliverable

Everything required by Section E ("Final Reports") of the CS445 project
description for the 18 May 2026 final submission. Replace `GroupXX` and the
three `Member 1/2/3` placeholders with the real values before submitting.

## What this is

A genuinely working system for **AmbiStory** graded word-sense plausibility
rating: a **3-component calibrated ensemble** —

* **A** — zero-shot NLI sense-plausibility scorer (no training)
* **B** — fine-tuned gloss-informed encoder regressor
* **C** — *novel* Likert-distribution head (predicts the 5-vote distribution,
  reads out a continuous expected rating + a free uncertainty)

blended with weights tuned on a **homonym-disjoint train hold-out** (the dev /
test splits are homonym-disjoint, so dev is report-only). Open-source
HuggingFace models only — **no API keys, fully reproducible**.

## Contents

| Path | What it is |
|------|------------|
| `GroupXX_final_report.{md,pdf,docx}` | Report (md is the source; pdf/docx are built from it, numbers injected from `results.json`). |
| `GroupXX_final_presentation.pptx` | ~16-slide deck for the 15-minute talk, speaker notes included. |
| `GroupXX_final_notebook.ipynb` | Executed, **output logs preserved** — narrated runnable system. |
| `src/*.py` | The system: `data, metrics, nli_scorer, encoder_reg, likert_head, ensemble, run_all, report_fill`. |
| `_build_{notebook,pdf,docx,pptx}.py`, `_make_figures.py` | Deliverable generators. |
| `_check_submission.py` | Validates a predictions file with the official `format_check`. |
| `_make_archive.py` | Bundles everything into `GroupXX_final_submission.zip`. |
| `figures/*.png` | Confusion matrix, per-class F1, macro P/R, PR curves, calibration, training curves, ablations, SOTA comparison, uncertainty. |
| `predictions/*.jsonl` | Per-system dev predictions + `ensemble_test.jsonl` (the CodaBench file) + `_int` fallback. |
| `results.json` | Single source of truth — every reported number, written by `run_all.py`. |
| `requirements_final.txt` | Dependencies beyond the repo root `requirements.txt`. |

## Reproduce end-to-end

Run from the **repository root** (`semeval26-05-scripts/`):

```bash
python -m pip install -r requirements.txt
python -m pip install -r final/requirements_final.txt
# GPU machines: install a CUDA torch build first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# 1. Train + evaluate everything → results.json, predictions/, scored by the
#    OFFICIAL scoring.py. GPU is auto-detected.
python final/src/run_all.py
#    Runtime modes (auto):
#      GPU            → DeBERTa-v3-large + DeBERTa-v3-large-MNLI, 3 seeds,
#                        4 epochs, mean-pool + rank-aligned loss + LLRD
#                        (canonical; ~2–3.5 h)
#      FAST=1 (GPU)   → RoBERTa-base, 2 epochs               (~45 min)
#      no GPU         → RoBERTa-base + DeBERTa-v3-base-MNLI, full data (weaker; CPU)
#      SKIP_HEAVY=1   → lexical baselines only (structural CI, no downloads)

# 2. Build EVERYTHING from those artifacts in one command:
#    figures → PDF → DOCX → PPTX → executed notebook → submission check → zip
python final/_finalize.py
#    (equivalently, run the steps individually:)
#      python final/_make_figures.py
#      python final/_build_pdf.py
#      python final/_build_docx.py
#      python final/_build_pptx.py
#      python final/_build_notebook.py     # builds AND executes (keeps outputs)
#      python final/_check_submission.py final/predictions/ensemble_test.jsonl
#      python final/_make_archive.py        # → final/GroupXX_final_submission.zip
```

`run_all.py` writes `final/_solution_dev.jsonl` and self-scores with the repo's
real `scoring.py`, so all reported numbers are produced by the grader's own
code. The notebook loads these artifacts and re-scores them live; set
`RUN_FULL=1` to retrain inside the notebook.

## CodaBench submission

The competition file is **`final/predictions/ensemble_test.jsonl`** — 930 lines,
string ids, each exactly once. Predictions are **continuous** values in [1,5]:
`scoring.py` consumes the raw value for both metrics, which materially improves
Spearman (`format_check` only int-casts for a non-fatal warning). A strict
integer fallback (`ensemble_test_int.jsonl`) is provided. Zip the chosen file
and upload it on the task's "My Submissions" tab.

## Notes on reproducibility & honesty

* Global `SEED = 42`; deterministic dataloaders; HF cache under `final/.hf_cache`.
* Every tuning decision (early stopping, NLI calibration, ensemble weights,
  final calibration) uses the **homonym-disjoint hold-out**, never dev.
* The included `results.json` records the exact runtime mode that produced it.
  On a CPU-only machine the numbers are below the canonical GPU configuration;
  re-run on a GPU and rebuild to refresh the report/slides automatically.
* Reused: official `scoring.py`/`format_check.py`, HuggingFace `transformers`,
  PyTorch, scikit-learn, public pre-trained checkpoints. **Our contribution:**
  the novel Likert-distribution head and uncertainty gate, the homonym-disjoint
  evaluation protocol, the gloss-informed input, the NLI sense-plausibility
  formulation + calibration, the ensemble, and all glue/figures/reporting.
