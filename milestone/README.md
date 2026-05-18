# GroupXX — SemEval-2026 Task 5 Milestone Deliverables

Everything required by Section E.1 of the CS445 project description for the
13 April 2026 milestone lives in this folder. Replace `GroupXX` with your real
group number and the three `Member 1/2/3` placeholders with the actual team
names before submission.

## Contents

| File | What it is |
|------|------------|
| `GroupXX_milestone_report.md` | The source of the milestone report (Markdown). Edit this, then re-run `_build_pdf.py`. |
| `GroupXX_milestone_report.pdf` | Rendered 3-page PDF (≤4-page limit satisfied). |
| `GroupXX_milestone_presentation.pptx` | 8-slide deck targeted at the 5-minute talk; speaker notes included. |
| `GroupXX_milestone_notebook.ipynb` | Executed Jupyter notebook — data loading, EDA, official-metric evaluation, baselines, prompt/encoder scaffolds. Output logs preserved as rubric requires. |
| `figures/*.png` | EDA and baseline-result plots used by the notebook and the slide deck. |
| `baseline_scores.json` | Numeric results for every baseline on dev. |
| `predictions/*.jsonl` | JSONL baseline predictions in the submission format. |
| `requirements_milestone.txt` | Extra Python dependencies beyond the repo's `requirements.txt`. |
| `_eda_and_baselines.py`, `_build_notebook.py`, `_build_pdf.py`, `_build_pptx.py` | Generator scripts — re-run any of them after editing the source. |

## Reproducing end-to-end

From the repository root:

```
python -m pip install -r milestone/requirements_milestone.txt

# EDA + baseline evaluation, regenerates figures + baseline_scores.json
python milestone/_eda_and_baselines.py

# Rebuild the notebook AND execute it so outputs are embedded
python milestone/_build_notebook.py

# Re-render the report PDF from the Markdown source
python milestone/_build_pdf.py

# Rebuild the slide deck
python milestone/_build_pptx.py
```

## Milestone rubric coverage

Section E.1 of the description asks for four report sections and a ≤4-page
limit:

* **1. Introduction** — task description + what has been done so far. → `GroupXX_milestone_report.md` §1.
* **2. Dataset Selection** — dataset, splits, EDA. → §2 + the notebook §1–§2 (rating histograms, σ distribution, top-20 homonyms, length statistics, cross-split homonym overlap).
* **3. Approach Plan** — methodology draft with ≥3 cited papers. → §3 cites five papers from allowed venues (NAACL, NeurIPS, ACL × 2, NAACL).
* **4. Next Steps** — workload division. → §4 with a per-week × per-member table through 18 May 2026.

The talk is 5 minutes + 5 minutes Q & A. The deck has 8 slides following the
same section order (intro, task, dataset, EDA, related work, approach, workload
+ baselines, next steps).

## Files not to modify

The generator scripts (`_*.py`) are helpers. The *deliverables* are the four
`GroupXX_*` files at the top of the table — those are what you upload.
