"""Ensemble: combine Components A/B/C into the final submission.

All three components emit a continuous score in [1, 5] for every fold
(hold-out, dev, test).  We:

1.  grid-search a convex weight vector ``(w_A, w_B, w_C)`` that maximises
    **Spearman on the homonym-disjoint train hold-out** (mirrors the dev/test
    regime — dev is never used for tuning);
2.  apply one final isotonic calibration ``ensemble → average`` fit on the
    hold-out and clip to [1, 5];
3.  optionally down-weight Component C where its predicted variance is high
    (the "uncertainty-gated" variant — an ablation, not the default);
4.  write per-system and ensemble predictions, plus the final CodaBench file
    as **continuous** values (the official scorer consumes the raw number) with
    a rounded-integer fallback alongside.
"""
from __future__ import annotations

import itertools

import numpy as np

from .data import write_predictions_jsonl
from .metrics import spearman
from .nli_scorer import Calibrator


def _weight_grid(step: float = 0.1):
    """All convex 3-component weights on a ``step`` grid (sum = 1)."""
    pts = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    for wa, wb in itertools.product(pts, pts):
        wc = round(1.0 - wa - wb, 4)
        if -1e-9 <= wc <= 1.0 + 1e-9:
            yield (float(wa), float(wb), max(0.0, wc))


def search_weights(
    holdout_preds: dict, holdout_avg, margin: float = 0.01
) -> tuple[float, float, float]:
    """Pick a convex ``(w_A,w_B,w_C)`` that robustly maximises hold-out Spearman.

    The hold-out is small (~276 homonym-disjoint rows) and the 66-point convex
    grid overfits it easily — empirically the grid-optimal blend has been
    *worse* on dev than the single best component. Two guards:

    1. **No-worse-than-best-single.** Compute each component's solo hold-out ρ;
       only trust a blend if it beats the best single component by ``margin``,
       otherwise fall back to that component's one-hot weight.
    2. **Top-K stabilisation.** Average the 5 best grid weight vectors instead
       of taking the single argmax; if that averaged blend does not itself
       clear the margin, fall back to the best single component as well.
    """
    a, b, c = holdout_preds["A"], holdout_preds["B"], holdout_preds["C"]
    singles = {
        (1.0, 0.0, 0.0): spearman(a, holdout_avg),
        (0.0, 1.0, 0.0): spearman(b, holdout_avg),
        (0.0, 0.0, 1.0): spearman(c, holdout_avg),
    }
    best_single_w = max(singles, key=singles.get)
    best_single_rho = singles[best_single_w]

    scored = []
    best_w, best_rho = (1 / 3, 1 / 3, 1 / 3), -2.0
    for wa, wb, wc in _weight_grid():
        rho = spearman(wa * a + wb * b + wc * c, holdout_avg)
        scored.append(((wa, wb, wc), rho))
        if rho > best_rho:
            best_rho, best_w = rho, (wa, wb, wc)

    # Guard 1: the best blend must clear the best single component.
    if best_rho < best_single_rho + margin:
        return best_single_w

    # Guard 2: stabilise by averaging the top-5 weight vectors, but only keep it
    # if it still clears the margin (averaging weights does not preserve the
    # maximisation); otherwise fall back to the best single component.
    scored.sort(key=lambda x: x[1], reverse=True)
    topk = np.array([w for w, _ in scored[:5]], dtype=float)
    avg_w = topk.mean(axis=0)
    avg_w = avg_w / avg_w.sum()
    avg_rho = spearman(
        avg_w[0] * a + avg_w[1] * b + avg_w[2] * c, holdout_avg
    )
    if avg_rho >= best_single_rho + margin:
        return (float(avg_w[0]), float(avg_w[1]), float(avg_w[2]))
    if best_rho >= best_single_rho + margin:
        return best_w
    return best_single_w


def uncertainty_gate(preds: dict, var_C, w):
    """Ablation: shrink C's weight per-sample when its variance is high.

    Returns a blended vector where ``w_C`` is scaled by ``1/(1+var)`` (re-
    normalised per row) — high predicted spread ⇒ trust the supervised
    regressor and the NLI prior more.
    """
    wa, wb, wc = w
    g = 1.0 / (1.0 + np.asarray(var_C, float))
    wc_i = wc * g
    norm = wa + wb + wc_i
    return (wa * preds["A"] + wb * preds["B"] + wc_i * preds["C"]) / norm


class Ensemble:
    """Holds the tuned weights + final calibrator."""

    def __init__(self, calib_method: str = "isotonic"):
        self.weights = None
        self.calibrator = Calibrator(calib_method)

    def fit(self, holdout_preds: dict, holdout_avg) -> "Ensemble":
        self.weights = search_weights(holdout_preds, holdout_avg)
        wa, wb, wc = self.weights
        blend = (
            wa * holdout_preds["A"]
            + wb * holdout_preds["B"]
            + wc * holdout_preds["C"]
        )
        self.calibrator.fit(blend, np.asarray(holdout_avg, float))
        return self

    def predict(self, preds: dict) -> np.ndarray:
        wa, wb, wc = self.weights
        blend = wa * preds["A"] + wb * preds["B"] + wc * preds["C"]
        return self.calibrator.predict(blend)


def write_all_predictions(ids_dev, ids_test, dev_preds: dict, test_preds: dict,
                          ensemble_dev, ensemble_test, preds_dir):
    """Emit every per-system dev file + the final ensemble dev/test files.

    The CodaBench submission is the *continuous* ``ensemble_test.jsonl``; an
    integer fallback (``ensemble_test_int.jsonl``) is written next to it for the
    strict-format variant and the continuous-vs-int ablation.
    """
    preds_dir.mkdir(parents=True, exist_ok=True)
    name = {"A": "nli", "B": "encoder", "C": "likert"}
    for key, fname in name.items():
        write_predictions_jsonl(
            ids_dev, dev_preds[key], preds_dir / f"{fname}_dev.jsonl"
        )
    write_predictions_jsonl(ids_dev, ensemble_dev, preds_dir / "ensemble_dev.jsonl")
    write_predictions_jsonl(ids_test, ensemble_test, preds_dir / "ensemble_test.jsonl")
    write_predictions_jsonl(
        ids_test, ensemble_test, preds_dir / "ensemble_test_int.jsonl", as_int=True
    )
