"""Programmatically build Group24_final_notebook.ipynb and execute it so cell
outputs (EDA, official scores, figures, ablation tables, logs) are embedded —
the rubric requires preserving output logs.

The notebook is the narrated system: it imports the same ``src`` package the
pipeline uses, shows the data/EDA, then **re-scores the saved predictions with
the official scoring.py live** (fast and genuine) and renders every figure and
table.  A clearly-marked final cell re-runs the full training pipeline; it is
guarded by ``RUN_FULL=1`` so executing the notebook for output capture does not
trigger a multi-hour retrain (the canonical GPU run is ``src/run_all.py`` and
is documented in README).  Execution uses ``jupyter nbconvert --execute``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

FINAL = Path(__file__).resolve().parent
OUT = FINAL / "Group24_final_notebook.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""# Group 24 — SemEval-2026 Task 5 · Final Notebook
### Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding

**Members:** Taylan İrak · Teammate 2 (name) · Teammate 3 (name)  ·  **Date:** 18 May 2026

This notebook is the narrated, runnable version of our final system. It:

1. loads the **AmbiStory** data and recaps the EDA that motivates the design;
2. explains and runs the **3-component ensemble**
   (A: zero-shot NLI · B: gloss-informed regressor · C: novel
   Likert-distribution head);
3. scores every system with the **official `scoring.py`** and shows the
   confusion matrix, per-class F1, macro precision/recall, PR curves,
   calibration, ablations and the SOTA/baseline comparison.

> **Reproducibility.** All code lives in `final/src/` and is imported here (not
> re-pasted) so the notebook and the command-line pipeline cannot diverge. The
> canonical end-to-end run is `python final/src/run_all.py` (GPU →
> DeBERTa-v3-large, 3 seeds; CPU → RoBERTa-base fallback). To keep this
> notebook's execution bounded, the training cell is guarded by `RUN_FULL=1`;
> by default the notebook **loads the artifacts produced by `run_all.py`** and
> re-scores them live with the official scorer. Output logs are preserved as
> the rubric requires.
""")

code("""import json, os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

FINAL = Path.cwd()
if FINAL.name != "final":
    FINAL = FINAL / "final"          # allow running from the repo root
REPO = FINAL.parent
sys.path.insert(0, str(FINAL))
print("repo :", REPO)
print("final:", FINAL)""")

md("""## 1. Task and data

Each instance is a 5-sentence story with a target homonym plus one candidate
sense; five annotators rate plausibility 1–5. We predict the human mean and are
scored by **Spearman ρ** and **accuracy@std**. Two facts shape everything:
the target is *graded with real disagreement*, and the splits are *near
homonym-disjoint* (dev ∩ train homonyms = 0) so lexical memorisation cannot
transfer and **dev is report-only** — all tuning uses a homonym-disjoint
hold-out carved from train.""")

code("""from src import data as D
train, dev, test = D.load_split("train"), D.load_split("dev"), D.load_split("test")
print("sizes:", {"train": len(train), "dev": len(dev), "test": len(test)})

hom = lambda df: set(df["homonym"].astype(str))
print("dev ∩ train homonyms :", len(hom(dev)  & hom(train)))
print("test ∩ train homonyms:", len(hom(test) & hom(train)))
fit_mask, hold_mask = D.make_holdout(train, frac=0.12, seed=42)
print("homonym-disjoint hold-out:",
      {"fit": int(fit_mask.sum()), "hold": int(hold_mask.sum()),
       "hold homonyms": len(hom(train[hold_mask]))})
display(train[["homonym","judged_meaning","average","stdev","choices"]].head(3))""")

code("""# EDA recap — graded target with substantial annotator disagreement
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(train["average"], bins=np.linspace(1,5,21), color="#0B3D91", alpha=.85)
ax[0].set_title(f"train avg (mean={train['average'].mean():.2f})"); ax[0].set_xlabel("average")
ax[1].hist(train["stdev"], bins=25, color="teal", alpha=.85)
ax[1].axvline(1.2, color="red", ls="--",
              label=f"σ≥1.2: {(train['stdev']>=1.2).mean():.0%}")
ax[1].set_title("annotator disagreement σ"); ax[1].legend(); plt.tight_layout(); plt.show()""")

md("""## 2. The system (imported from `final/src/`)

* **Component A — zero-shot NLI** (`src/nli_scorer.py`): premise = story,
  hypothesis = *"In this story, \"<homonym>\" means <gloss>."*; signal
  `P(entail) − P(contradiction)`, isotonic-calibrated to [1,5]. No training.
* **Component B — gloss-informed regressor** (`src/encoder_reg.py`): encoder +
  linear head, MSE on the human mean, early-stopped on the hold-out.
* **Component C — novel Likert-distribution head** (`src/likert_head.py`):
  predicts the 5-vote distribution, reads out `E[k]` (continuous) and a free
  uncertainty `Var[k]`.
* **Ensemble** (`src/ensemble.py`): convex weights tuned on the hold-out +
  isotonic calibration. Submission is **continuous** (the official scorer reads
  the raw value — a key, documented finding).""")

code("""# Show the actual implementation is the source of truth (no re-paste).
import inspect
from src import nli_scorer, likert_head
print(inspect.getsource(nli_scorer.Calibrator))
print("--- novel loss (Component C): KL(votes||p) + 0.3*MSE(E[k], mean) ---")
print("".join(inspect.getsourcelines(likert_head._train_one_seed)[0][:1]))""")

md("""## 3. Run the pipeline (guarded) or load its artifacts

Set the environment variable `RUN_FULL=1` to retrain end-to-end (GPU
recommended; ~hours on CPU). Otherwise we load `results.json` /
`predictions/*.jsonl` produced by `python final/src/run_all.py` and re-score
them **live with the official `scoring.py`** so the numbers below are genuine
and reproducible.""")

code("""if os.environ.get("RUN_FULL") == "1":
    print("RUN_FULL=1 → executing the full pipeline (this can take hours)…")
    subprocess.run([sys.executable, str(FINAL/"src"/"run_all.py")], check=True)

res_path = FINAL / "results.json"
assert res_path.exists(), ("results.json not found — run "
    "`python final/src/run_all.py` first (or set RUN_FULL=1).")
RES = json.loads(res_path.read_text(encoding="utf-8"))
print("runtime that produced these results:", RES["meta"]["runtime_mode"])
print("data:", RES["meta"]["sizes"], "| hold-out:", RES["meta"]["holdout"])""")

code("""# Re-score the saved ENSEMBLE dev predictions with the OFFICIAL scoring.py
from src import metrics as M
sol = D.write_solution_jsonl(dev, FINAL / "_solution_dev.jsonl")
official = M.official_score(sol, FINAL/"predictions"/"ensemble_dev.jsonl",
                            FINAL/"_nb_scores.json")
print("OFFICIAL ensemble dev:", official)""")

code("""# Headline comparison table (official scorer numbers from results.json)
rows = []
for k,v in RES.get("baselines",{}).items():
    rows.append([k, v["accuracy_at_std"], v.get("spearman")])
for key,lab in [("A_nli","A — zero-shot NLI"),
                ("B_encoder","B — gloss regressor"),
                ("C_likert","C — Likert head (novel)")]:
    c = RES["components"].get(key,{})
    if "dev" in c: rows.append([lab, c["dev"]["accuracy_at_std"], c["dev"]["spearman"]])
e = RES["ensemble"].get("dev_continuous",{})
rows.append(["ENSEMBLE (continuous)", e.get("accuracy_at_std"), e.get("spearman")])
ei = RES["ensemble"].get("dev_integer",{})
rows.append(["ensemble (rounded int)", ei.get("accuracy_at_std"), ei.get("spearman")])
tbl = pd.DataFrame(rows, columns=["system","accuracy@std","Spearman ρ"])
print("verdict vs rubric:", RES["ensemble"].get("verdict"))
display(tbl.style.format({"accuracy@std":"{:.4f}","Spearman ρ":"{:+.4f}"}))""")

code("""# Ablations — the journey (continuous-vs-int is the headline lever)
print(json.dumps(RES.get("ablations",{}), indent=2))""")

md("""## 4. Figures (regenerated from the saved predictions)

Confusion matrix, per-class F1, macro precision/recall, one-vs-rest PR curves,
calibration, training curves, ablations, SOTA/baseline comparison, and the
novelty's uncertainty-vs-error plot.""")

code("""subprocess.run([sys.executable, str(FINAL/"_make_figures.py")], check=True)
for f in ["fig9_sota_baseline_comparison.png","fig2_confusion_matrix.png",
          "fig3_per_class_f1.png","fig4_macro_PR.png","fig5_pr_curves.png",
          "fig6_calibration.png","fig7_training_curves.png",
          "fig8_ablations.png","fig10_uncertainty_vs_error.png"]:
    p = FINAL/"figures"/f
    if p.exists(): display(Image(filename=str(p)))""")

code("""# Validate the FINAL CodaBench submission file
subprocess.run([sys.executable, str(FINAL/"_check_submission.py"),
                str(FINAL/"predictions"/"ensemble_test.jsonl")], check=True)""")

md("""## 5. Pipeline run log

Captured stdout of `python final/src/run_all.py` — preserved as required.""")

code("""log = FINAL / "_run_all.log"
if log.exists():
    # The training log contains console progress bytes / non-UTF-8 chars on
    # Windows; decode leniently and strip carriage returns for readability.
    txt = log.read_text(encoding="utf-8", errors="replace").replace("\\r", "\\n")
    print(txt[-6000:])
else:
    print("(_run_all.log not present — run src/run_all.py to generate it)")""")

md("""## 6. Conclusion

A 3-component ensemble (training-free NLI prior + gloss-informed regressor +
**novel Likert-distribution head**) with hold-out-tuned weights and continuous
calibrated output. It clears the rubric's OK bar on dev and improves Spearman
several-fold over the milestone lexical baselines, fully reproducibly from
`final/src/run_all.py`. Individual contributions are listed in the report §7.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", OUT, "— executing to embed outputs…")

rc = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", "--inplace", "--ExecutePreprocessor.timeout=3600",
     str(OUT)],
    cwd=str(FINAL.parent),
).returncode
print("notebook executed OK" if rc == 0 else
      f"WARNING: nbconvert returned {rc} — open the notebook to inspect")
