"""Evaluation metrics for SemEval-2026 Task 5.

Two entry points:

* :func:`evaluate` — fast in-process Spearman + accuracy@std, used inside
  training loops and the ensemble search.  It re-implements the *exact* logic of
  the official ``scoring.py`` (``is_within_standard_deviation``: a prediction is
  correct if it lies strictly inside ``avg ± stdev`` **or** within absolute
  distance ``< 1`` of the mean).
* :func:`official_score` — shells out to the repository's real ``scoring.py``
  against a generated ``solution.jsonl`` so the headline numbers in the report
  are produced by the grader's own code, not by us.

The classification-style metrics (confusion matrix, macro precision/recall,
per-class F1, one-vs-rest PR curves) are computed on the **rounded** prediction
vs. the **rounded human average** so that the rubric-mandated plots
(confusion matrix / F1 / macro P-R / PR curves) are well defined on a task
whose native target is continuous.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]


def accuracy_at_std(preds, avgs, stds) -> float:
    """Fraction correct under the official ``accuracy@std`` rule."""
    preds = np.asarray(preds, dtype=float)
    avgs = np.asarray(avgs, dtype=float)
    stds = np.asarray(stds, dtype=float)
    within_sd = (avgs - stds < preds) & (preds < avgs + stds)
    within_1 = np.abs(avgs - preds) < 1.0
    return float(np.mean(within_sd | within_1))


def spearman(preds, avgs) -> float:
    """Spearman ρ between predictions and human averages (0.0 if degenerate)."""
    if len(set(np.round(np.asarray(preds), 9))) < 2:
        return 0.0  # spearmanr is undefined for a constant vector
    rho, _ = spearmanr(np.asarray(preds, float), np.asarray(avgs, float))
    return 0.0 if np.isnan(rho) else float(rho)


def evaluate(preds, avgs, stds) -> dict:
    """Both official metrics in one call."""
    return {
        "spearman": spearman(preds, avgs),
        "accuracy_at_std": accuracy_at_std(preds, avgs, stds),
    }


def classification_metrics(preds, avgs) -> dict:
    """Confusion matrix + macro precision/recall + per-class F1.

    Predictions and gold averages are rounded to the nearest Likert integer
    (clipped to 1..5) so the task can also be reported as 5-way classification,
    as the project rubric requires.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = np.clip(np.round(np.asarray(preds, float)), 1, 5).astype(int)
    y_true = np.clip(np.round(np.asarray(avgs, float)), 1, 5).astype(int)
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
    }


def official_score(solution_jsonl: Path, predictions_jsonl: Path, out_json: Path) -> dict:
    """Run the repository's real ``scoring.py`` and return its scores.

    This guarantees the reported accuracy@std / Spearman are computed by the
    exact code the competition / grader uses.  Falls back to the in-process
    implementation only if the official script cannot be invoked, and labels
    that fallback explicitly.
    """
    solution_jsonl = Path(solution_jsonl)
    predictions_jsonl = Path(predictions_jsonl)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scoring.py"),
        str(solution_jsonl),
        str(predictions_jsonl),
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    print(proc.stdout)
    if proc.returncode != 0 or not out_json.exists():
        print(proc.stderr)
        raise RuntimeError(
            f"official scoring.py failed (rc={proc.returncode}); see output above"
        )
    with open(out_json, encoding="utf-8") as fh:
        scores = json.load(fh)
    scores["_source"] = "official scoring.py"
    return scores
