"""Generate every figure required by the final report / slides.

Reads the single source of truth ``final/results.json`` plus the per-system dev
prediction files, and writes ``final/figures/fig1..fig10 *.png`` (matplotlib
Agg, dpi 120 — same style as the milestone).  Safe to re-run after any
``run_all.py`` execution; missing/failed components are skipped gracefully so a
partial run still yields the figures it can.

Rubric coverage: confusion matrix (fig2), per-class F1 (fig3), macro
precision/recall (fig4), one-vs-rest precision-recall curves (fig5), plus
calibration, training curves, ablations, SOTA/baseline comparison and the
novelty's uncertainty-vs-error plot.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent
FIG = FINAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY = "#0B3D91"
OK_ACC, OK_RHO, GOOD_ACC, GOOD_RHO = 0.70, 0.60, 0.80, 0.70


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.relative_to(REPO)}")


def _load_results() -> dict:
    return json.loads((FINAL / "results.json").read_text(encoding="utf-8"))


def _load_preds(name) -> dict[str, float] | None:
    p = FINAL / "predictions" / name
    if not p.exists():
        return None
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[str(d["id"])] = float(d["prediction"])
    return out


def _dev_gold():
    raw = json.loads((REPO / "data" / "dev.json").read_text(encoding="utf-8"))
    ids = list(raw.keys())
    avg = np.array([float(raw[i]["average"]) for i in ids])
    std = np.array([float(raw[i]["stdev"]) for i in ids])
    return ids, avg, std


def _aligned(pred_map, ids):
    return np.array([pred_map[i] for i in ids])


# --------------------------------------------------------------------------- #
def fig1_eda():
    train = json.loads((REPO / "data" / "train.json").read_text(encoding="utf-8"))
    avg = np.array([v["average"] for v in train.values()])
    std = np.array([v["stdev"] for v in train.values()])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].hist(avg, bins=np.linspace(1, 5, 21), color=NAVY, alpha=0.85)
    ax[0].axvline(avg.mean(), color="red", ls="--", label=f"mean={avg.mean():.2f}")
    ax[0].set_title("Train: human-average rating")
    ax[0].set_xlabel("average"); ax[0].legend()
    ax[1].hist(std, bins=25, color="teal", alpha=0.85)
    ax[1].axvline(1.2, color="red", ls="--",
                  label=f"σ≥1.2: {(std >= 1.2).mean():.0%}")
    ax[1].set_title("Train: annotator disagreement σ")
    ax[1].set_xlabel("stdev"); ax[1].legend()
    _save(fig, "fig1_label_eda.png")


def fig2_confusion(res):
    cm = np.array(res["classification"]["confusion_matrix"])
    labels = res["classification"]["labels"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted (rounded)"); ax.set_ylabel("Gold (rounded avg)")
    ax.set_title("Ensemble — dev confusion matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, "fig2_confusion_matrix.png")


def fig3_per_class_f1(res):
    f1 = res["classification"]["per_class_f1"]
    labels = res["classification"]["labels"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar([str(x) for x in labels], f1, color=NAVY, alpha=0.85)
    ax.axhline(res["classification"]["macro_f1"], color="red", ls="--",
               label=f"macro-F1={res['classification']['macro_f1']:.3f}")
    ax.set_ylim(0, 1); ax.set_xlabel("Likert class")
    ax.set_ylabel("F1"); ax.set_title("Ensemble — per-class F1 (dev)")
    ax.legend()
    _save(fig, "fig3_per_class_f1.png")


def fig4_macro_pr(res):
    c = res["classification"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    vals = [c["macro_precision"], c["macro_recall"], c["macro_f1"]]
    ax.bar(["macro\nprecision", "macro\nrecall", "macro\nF1"], vals,
           color=["steelblue", "darkorange", NAVY], alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Ensemble — macro metrics (dev, 5-way)")
    _save(fig, "fig4_macro_PR.png")


def fig5_pr_curves(ids, avg):
    """One-vs-rest PR curves built from the continuous prediction.

    Score for class k = −(pred − k)²  (closer prediction ⇒ higher score), gold
    class = round(avg).  A standard, documented construction for turning a
    scalar regressor into per-class PR curves.
    """
    from sklearn.metrics import auc, precision_recall_curve

    pm = _load_preds("ensemble_dev.jsonl")
    if pm is None:
        return
    pred = _aligned(pm, ids)
    y = np.clip(np.round(avg), 1, 5).astype(int)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for k in range(1, 6):
        yk = (y == k).astype(int)
        if yk.sum() == 0:
            continue
        score = -((pred - k) ** 2)
        p, r, _ = precision_recall_curve(yk, score)
        ax.plot(r, p, label=f"class {k} (AUC={auc(r, p):.2f}, n={yk.sum()})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Ensemble — one-vs-rest PR curves (dev)")
    ax.legend(fontsize=8)
    _save(fig, "fig5_pr_curves.png")


def fig6_calibration(ids, avg):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, fname, col in [("Ensemble", "ensemble_dev.jsonl", NAVY),
                             ("NLI (A)", "nli_dev.jsonl", "darkorange"),
                             ("Encoder (B)", "encoder_dev.jsonl", "green"),
                             ("Likert (C)", "likert_dev.jsonl", "purple")]:
        pm = _load_preds(fname)
        if pm is None:
            continue
        pred = _aligned(pm, ids)
        bins = np.linspace(1, 5, 9)
        idx = np.digitize(pred, bins)
        xs, ys = [], []
        for b in range(1, len(bins) + 1):
            m = idx == b
            if m.sum() >= 5:
                xs.append(pred[m].mean()); ys.append(avg[m].mean())
        ax.plot(xs, ys, "o-", color=col, label=name, ms=4)
    ax.plot([1, 5], [1, 5], "k:", label="ideal")
    ax.set_xlabel("Mean predicted rating"); ax.set_ylabel("Mean gold average")
    ax.set_title("Calibration reliability (dev)"); ax.legend(fontsize=8)
    _save(fig, "fig6_calibration.png")


def fig7_training_curves(res):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    plotted = False
    for key, lab, col in [("B_encoder", "Encoder (B)", "green"),
                          ("C_likert", "Likert head (C)", "purple")]:
        comp = res["components"].get(key, {})
        for si, curve in enumerate(comp.get("holdout_curves", [])):
            ax.plot(range(1, len(curve) + 1), curve, "o-", color=col, alpha=0.7,
                    label=lab if si == 0 else None)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hold-out Spearman ρ")
    ax.set_title("Training curves (homonym-disjoint hold-out)")
    ax.legend(fontsize=8)
    _save(fig, "fig7_training_curves.png")


def fig8_ablations(res):
    ab = res.get("ablations", {})
    panels = []
    if "continuous_vs_int" in ab:
        cv = ab["continuous_vs_int"]
        panels.append(("Continuous vs Int\n(Spearman)",
                       {"continuous": cv["continuous"]["spearman"],
                        "rounded int": cv["rounded_int"]["spearman"]}))
    if "nli_calibration" in ab:
        panels.append(("NLI calibration\n(dev Spearman)", ab["nli_calibration"]))
    if "head_objective" in ab:
        panels.append(("Head objective\n(dev Spearman)", ab["head_objective"]))
    if "ensemble_drop_one" in ab:
        panels.append(("Ensemble drop-one\n(dev Spearman)",
                       {k.replace("drop_", "no "): v["spearman"]
                        for k, v in ab["ensemble_drop_one"].items()}))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, d) in zip(axes, panels):
        ax.bar(list(d.keys()), list(d.values()), color=NAVY, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for i, v in enumerate(d.values()):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Ablations — showing the journey", fontweight="bold")
    _save(fig, "fig8_ablations.png")


def fig9_sota(res):
    rows = []
    for k, v in res.get("baselines", {}).items():
        rows.append((k, v["accuracy_at_std"], v.get("spearman") or 0.0))
    for key, lab in [("A_nli", "NLI (A)"), ("B_encoder", "Encoder (B)"),
                     ("C_likert", "Likert (C)")]:
        c = res["components"].get(key, {})
        if "dev" in c:
            rows.append((lab, c["dev"]["accuracy_at_std"], c["dev"]["spearman"]))
    if "dev_continuous" in res.get("ensemble", {}):
        e = res["ensemble"]["dev_continuous"]
        rows.append(("ENSEMBLE", e["accuracy_at_std"], e["spearman"]))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(rows)), 4.2))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="accuracy@std",
           color="steelblue")
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="Spearman ρ",
           color="darkorange")
    ax.axhline(OK_ACC, color="grey", ls=":", label="OK acc=0.70")
    ax.axhline(OK_RHO, color="grey", ls="--", label="OK ρ=0.60")
    ax.axhline(GOOD_ACC, color="green", ls=":", lw=1, label="Good acc=0.80")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right", fontsize=8)
    ax.set_title("Milestone baselines → final components → ensemble (dev)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig9_sota_baseline_comparison.png")


def fig10_uncertainty():
    """Novelty: Component C predicted variance vs absolute error (dev)."""
    var_path = FINAL / "predictions" / "_likert_dev_var.npy"
    pm = _load_preds("likert_dev.jsonl")
    if pm is None or not var_path.exists():
        return
    ids, avg, _ = _dev_gold()
    pred = _aligned(pm, ids)
    var = np.load(var_path)
    err = np.abs(pred - avg)
    order = np.argsort(var)
    q = np.array_split(order, 5)
    xs = [var[g].mean() for g in q]
    ys = [err[g].mean() for g in q]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(xs, ys, "o-", color=NAVY)
    ax.set_xlabel("Predicted variance (uncertainty), quintile mean")
    ax.set_ylabel("Mean |pred − gold|")
    ax.set_title("Novelty: Likert-head uncertainty tracks error (dev)")
    _save(fig, "fig10_uncertainty_vs_error.png")


def main():
    res = _load_results()
    ids, avg, _ = _dev_gold()
    fig1_eda()
    if res.get("classification"):
        fig2_confusion(res)
        fig3_per_class_f1(res)
        fig4_macro_pr(res)
    fig5_pr_curves(ids, avg)
    fig6_calibration(ids, avg)
    fig7_training_curves(res)
    fig8_ablations(res)
    fig9_sota(res)
    fig10_uncertainty()
    print("Figures complete.")


if __name__ == "__main__":
    main()
