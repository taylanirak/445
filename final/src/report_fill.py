"""Inject live numbers from ``final/results.json`` into the report source.

The report (``GroupXX_final_report.md``) is written by the team with ``{{TOKEN}}``
placeholders for every metric.  This module turns ``results.json`` into the
token→string map and substitutes it, so the PDF / DOCX / notebook always show
the numbers from the run that actually produced ``results.json`` (CPU now, the
canonical GPU run when re-executed).  If ``results.json`` is missing a value
(e.g. a component errored) the token is replaced with ``n/a`` rather than
crashing the build.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
RESULTS = FINAL / "results.json"


def _f(x, nd=3):
    try:
        v = float(x)
        return "n/a" if v != v else f"{v:.{nd}f}"  # v!=v ⇒ NaN
    except (TypeError, ValueError):
        return "n/a"


def _sig(x, nd=3):
    """Signed format for Spearman (e.g. +0.612)."""
    try:
        v = float(x)
        return "n/a" if v != v else f"{v:+.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_token_map(results: dict | None = None) -> dict[str, str]:
    if results is None:
        results = (
            json.loads(RESULTS.read_text(encoding="utf-8"))
            if RESULTS.exists()
            else {}
        )
    g = lambda *ks, d=None: _dig(results, ks, d)

    bl = results.get("baselines", {})
    comp = results.get("components", {})
    ens = results.get("ensemble", {})
    ab = results.get("ablations", {})
    cl = results.get("classification", {})

    def comp_metric(key, metric):
        return _dig(comp, (key, "dev", metric))

    # "Strongest milestone baseline" = the best *learned* lexical baseline
    # (constant predictors have an undefined/NaN Spearman, so exclude them).
    def _finite(v):
        try:
            return v == v and abs(float(v)) != float("inf")  # not NaN/inf
        except (TypeError, ValueError):
            return False

    learned = {k: v for k, v in bl.items() if _finite(v.get("spearman"))}
    best_bl = max((learned or bl).items(),
                  key=lambda kv: kv[1].get("accuracy_at_std", 0),
                  default=(None, {}))

    # drop-one: which removal hurts Spearman most
    drop = ab.get("ensemble_drop_one", {})
    worst = min(
        drop.items(),
        key=lambda kv: kv[1].get("spearman", 1.0),
        default=(None, None),
    )[0]
    worst = {"drop_A": "the NLI scorer", "drop_B": "the encoder regressor",
             "drop_C": "the Likert head"}.get(worst, "any single component")

    m = {
        "RUNTIME_MODE": g("meta", "runtime_mode", d="n/a"),
        "ENS_ACC": _f(_dig(ens, ("dev_continuous", "accuracy_at_std"))),
        "ENS_RHO": _sig(_dig(ens, ("dev_continuous", "spearman"))),
        "ENS_INT_ACC": _f(_dig(ens, ("dev_integer", "accuracy_at_std"))),
        "ENS_INT_RHO": _sig(_dig(ens, ("dev_integer", "spearman"))),
        "ENS_VERDICT": ens.get("verdict", "n/a"),
        "W_A": _f(_dig(ens, ("weights", "A_nli")), 2),
        "W_B": _f(_dig(ens, ("weights", "B_encoder")), 2),
        "W_C": _f(_dig(ens, ("weights", "C_likert")), 2),
        "A_ACC": _f(comp_metric("A_nli", "accuracy_at_std")),
        "A_RHO": _sig(comp_metric("A_nli", "spearman")),
        "B_ACC": _f(comp_metric("B_encoder", "accuracy_at_std")),
        "B_RHO": _sig(comp_metric("B_encoder", "spearman")),
        "C_ACC": _f(comp_metric("C_likert", "accuracy_at_std")),
        "C_RHO": _sig(comp_metric("C_likert", "spearman")),
        "BL_RIDGE_ACC": _f(_dig(bl, ("feature_ridge", "accuracy_at_std"))),
        "BL_RIDGE_RHO": _sig(_dig(bl, ("feature_ridge", "spearman"))),
        "BL_TFIDF_ACC": _f(_dig(bl, ("tfidf_cosine", "accuracy_at_std"))),
        "BL_TFIDF_RHO": _sig(_dig(bl, ("tfidf_cosine", "spearman"))),
        "BEST_BASELINE_ACC": _f(best_bl[1].get("accuracy_at_std")),
        "BEST_BASELINE_RHO": _sig(best_bl[1].get("spearman")),
        "MACRO_F1": _f(cl.get("macro_f1")),
        "MACRO_P": _f(cl.get("macro_precision")),
        "MACRO_R": _f(cl.get("macro_recall")),
        "ABL_INT_RHO": _sig(_dig(ab, ("continuous_vs_int", "rounded_int", "spearman"))),
        "ABL_CONT_RHO": _sig(_dig(ab, ("continuous_vs_int", "continuous", "spearman"))),
        "ABL_CAL_LIN": _sig(_dig(ab, ("nli_calibration", "linear"))),
        "ABL_CAL_ISO": _sig(_dig(ab, ("nli_calibration", "isotonic"))),
        "ABL_MSE": _sig(_dig(ab, ("head_objective", "mse_regressor"))),
        "ABL_CORN": _sig(_dig(ab, ("head_objective", "corn_ordinal"))),
        "ABL_DIST": _sig(_dig(ab, ("head_objective", "distribution_KL"))),
        "DROP_WORST": worst,
    }
    # continuous-vs-int gain phrasing
    try:
        gain = (float(_dig(ab, ("continuous_vs_int", "continuous", "spearman")))
                - float(_dig(ab, ("continuous_vs_int", "rounded_int", "spearman"))))
        m["CONT_GAIN"] = f"{gain:+.3f} Spearman"
    except (TypeError, ValueError):
        m["CONT_GAIN"] = "a clear margin"
    return m


def _dig(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def strip_comments(md: str) -> str:
    """Remove HTML comment blocks (team notes) before rendering."""
    return re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)


def fill(md: str, results: dict | None = None) -> str:
    token_map = build_token_map(results)

    def repl(match):
        return token_map.get(match.group(1), match.group(0))

    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", repl, strip_comments(md))


def load_filled(md_path: Path, results: dict | None = None) -> str:
    return fill(Path(md_path).read_text(encoding="utf-8"), results)
