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

import numpy as np

from .data import write_predictions_jsonl
from .metrics import spearman
from .nli_scorer import Calibrator


#: Small, *coarse* convex-weight candidate set. A 66-point grid overfits the
#: ~276-row homonym-disjoint hold-out (the grid-optimal blend was empirically
#: worse on dev than the best single component). A handful of robust mixtures —
#: the one-hots, a few B/C blends that drop the weak NLI prior, and an equal
#: triple — cannot overfit a 276-row signal yet still capture a genuine
#: complementary gain when one exists.
_WEIGHT_CANDIDATES = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),  # solo
    (0.0, 0.5, 0.5), (0.0, 0.6, 0.4), (0.0, 0.4, 0.6),  # B/C only (drop A)
    (0.0, 0.7, 0.3), (0.0, 0.3, 0.7),
    (0.2, 0.4, 0.4),                                     # small A prior
    (1 / 3, 1 / 3, 1 / 3),                               # equal triple
)


def search_weights(
    holdout_preds: dict, holdout_avg, margin: float = 0.01
) -> tuple[float, float, float]:
    """Pick a convex ``(w_A,w_B,w_C)`` that *robustly* maximises hold-out ρ.

    Instead of a fine grid (which overfits the tiny homonym-disjoint hold-out
    and produced a blend worse on dev than the best single component), we
    evaluate only the coarse :data:`_WEIGHT_CANDIDATES` set and keep the
    no-worse-than-best-single guard: the chosen mixture is adopted only if it
    beats the best single component's hold-out ρ by ``margin``; otherwise the
    system falls back to that single component. Both the candidate set and the
    guard make the selection insensitive to 276-row hold-out noise.
    """
    a, b, c = holdout_preds["A"], holdout_preds["B"], holdout_preds["C"]
    singles = {
        (1.0, 0.0, 0.0): spearman(a, holdout_avg),
        (0.0, 1.0, 0.0): spearman(b, holdout_avg),
        (0.0, 0.0, 1.0): spearman(c, holdout_avg),
    }
    best_single_w = max(singles, key=singles.get)
    best_single_rho = singles[best_single_w]

    best_w, best_rho = best_single_w, -2.0
    for wa, wb, wc in _WEIGHT_CANDIDATES:
        rho = spearman(wa * a + wb * b + wc * c, holdout_avg)
        if rho > best_rho:
            best_rho, best_w = rho, (wa, wb, wc)

    # Guard: a mixture must clear the best single component by a real margin,
    # otherwise the single component is the more reliable choice.
    if best_rho < best_single_rho + margin:
        return best_single_w
    return best_w


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
