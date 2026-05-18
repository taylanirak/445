"""EDA + baseline evaluation for SemEval-2026 Task 5 (AmbiStory).

Run from the repo root:
    python milestone/_eda_and_baselines.py

Generates figures under milestone/figures/ and writes
milestone/baseline_scores.json with Spearman + accuracy@std for each baseline.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIG = REPO / "milestone" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_split(name: str) -> pd.DataFrame:
    with open(DATA / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for sid, entry in raw.items():
        rows.append({"id": sid, **entry, "split": name})
    return pd.DataFrame(rows)


def _save(fig, fname):
    path = FIG / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO)}")


def eda(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("=== Dataset sizes ===")
    sizes = {"train": len(train), "dev": len(dev), "test": len(test)}
    for k, v in sizes.items():
        print(f"  {k}: {v}")

    print("=== Schema ===")
    print("  columns:", list(train.columns))

    print("=== Rating distribution (avg) ===")
    desc = train["average"].describe()
    print(desc)

    # Fig 1: histogram of average rating per split
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(1, 5, 21)
    for df, label in [(train, "train"), (dev, "dev"), (test, "test")]:
        ax.hist(df["average"], bins=bins, alpha=0.55, label=f"{label} (n={len(df)})")
    ax.set_xlabel("Human-averaged plausibility rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of average plausibility ratings")
    ax.legend()
    _save(fig, "fig1_avg_rating_hist.png")

    # Fig 2: histogram of stdev
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(train["stdev"], bins=25, color="steelblue", alpha=0.8)
    ax.axvline(train["stdev"].mean(), color="red", linestyle="--",
               label=f"mean σ={train['stdev'].mean():.2f}")
    ax.set_xlabel("Annotator standard deviation")
    ax.set_ylabel("Count (train)")
    ax.set_title("Inter-annotator disagreement (σ over 5 annotators)")
    ax.legend()
    _save(fig, "fig2_stdev_hist.png")

    # Fig 3: pooled discrete choices
    pooled = [c for row in train["choices"] for c in row]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = Counter(pooled)
    xs = sorted(vc)
    ax.bar(xs, [vc[x] for x in xs], color="darkorange")
    ax.set_xlabel("Individual Likert choice (1–5)")
    ax.set_ylabel("Count (train, 5 annotators × N)")
    ax.set_title("Pooled annotator choices on train")
    _save(fig, "fig3_choice_distribution.png")

    # Fig 4: top homonyms
    hom_train = Counter(train["homonym"])
    top20 = hom_train.most_common(20)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([h for h, _ in top20][::-1], [c for _, c in top20][::-1],
            color="teal")
    ax.set_xlabel("Count (train)")
    ax.set_title("Top-20 homonyms by sample frequency")
    _save(fig, "fig4_top_homonyms.png")

    # Fig 5: length distributions
    def toklen(s):
        return len(str(s).split())

    for split_df, split_name in [(train, "train")]:
        split_df["len_pre"] = split_df["precontext"].map(toklen)
        split_df["len_sent"] = split_df["sentence"].map(toklen)
        split_df["len_end"] = split_df["ending"].map(toklen)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(train["len_pre"], bins=30, alpha=0.55, label="precontext")
    ax.hist(train["len_sent"], bins=30, alpha=0.55, label="sentence")
    ax.hist(train["len_end"], bins=30, alpha=0.55, label="ending")
    ax.set_xlabel("# whitespace tokens")
    ax.set_ylabel("Count (train)")
    ax.set_title("Token lengths of narrative components")
    ax.legend()
    _save(fig, "fig5_length_hist.png")

    # Homonym overlap
    hom_tr = set(train["homonym"]); hom_dev = set(dev["homonym"]); hom_te = set(test["homonym"])
    overlap = {
        "train_unique": len(hom_tr),
        "dev_unique": len(hom_dev),
        "test_unique": len(hom_te),
        "dev_in_train": len(hom_dev & hom_tr),
        "test_in_train": len(hom_te & hom_tr),
    }
    print("=== Homonym overlap ===")
    for k, v in overlap.items():
        print(f"  {k}: {v}")

    # Nonsensical fraction — field may be list[bool]; use the first annotator flag aggregated
    def nonsensical_any(v):
        if isinstance(v, list) and len(v) > 0:
            return int(any(bool(x) for x in v))
        return int(bool(v))

    train["nonsensical_any"] = train["nonsensical"].apply(nonsensical_any)
    grouped = train.groupby(pd.cut(train["average"], bins=[0, 2, 3, 4, 5]))\
                    ["nonsensical_any"].mean()
    print("=== 'nonsensical' rate per rating bucket (train) ===")
    print(grouped)

    stats = {
        "sizes": sizes,
        "mean_avg_train": float(train["average"].mean()),
        "median_avg_train": float(train["average"].median()),
        "mean_stdev_train": float(train["stdev"].mean()),
        "unique_homonyms": overlap,
        "mean_tok_lengths": {
            "precontext": float(train["len_pre"].mean()),
            "sentence": float(train["len_sent"].mean()),
            "ending": float(train["len_end"].mean()),
        },
    }
    return stats


def write_predictions(pred_map: dict[str, float], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for sid, pred in pred_map.items():
            f.write(json.dumps({"id": str(sid), "prediction": int(round(float(pred)))}) + "\n")


def accuracy_at_std(preds: list[float], golds_avg: list[float], golds_std: list[float]) -> float:
    correct = 0
    for p, a, s in zip(preds, golds_avg, golds_std):
        within_sd = (a - s) < p < (a + s)
        within_1 = abs(a - p) < 1
        if within_sd or within_1:
            correct += 1
    return correct / len(preds)


def evaluate(preds: list[float], avgs: list[float], stds: list[float]) -> dict:
    corr, _ = spearmanr(preds, avgs)
    acc = accuracy_at_std(preds, avgs, stds)
    return {"spearman": float(corr) if not np.isnan(corr) else 0.0,
            "accuracy_at_std": float(acc)}


def main():
    print("Loading splits...")
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")

    print("\n--- EDA ---")
    stats = eda(train, dev, test)

    # Baselines on dev set
    dev_avg = dev["average"].tolist()
    dev_std = dev["stdev"].tolist()
    dev_ids = dev["id"].tolist()

    print("\n--- Baselines on dev ---")
    results = {}

    # Majority (prediction=4, as in repo script)
    majority_preds = [4] * len(dev)
    results["majority_4"] = evaluate(majority_preds, dev_avg, dev_std)

    # Random in [1,5]
    rng = random.Random(SEED)
    random_preds = [rng.randint(1, 5) for _ in dev_ids]
    results["random"] = evaluate(random_preds, dev_avg, dev_std)

    # Mean-of-train rounded
    mean_pred = int(round(train["average"].mean()))
    mean_preds = [mean_pred] * len(dev)
    results[f"mean_of_train_rounded_{mean_pred}"] = evaluate(mean_preds, dev_avg, dev_std)

    # Per-homonym train mean (falls back to global mean if unseen)
    hom_mean = train.groupby("homonym")["average"].mean().to_dict()
    fallback = train["average"].mean()
    hom_preds = [int(round(hom_mean.get(h, fallback))) for h in dev["homonym"]]
    results["per_homonym_mean"] = evaluate(hom_preds, dev_avg, dev_std)

    # Sense-definition-length heuristic: longer gloss => more specific => likely lower plausibility? sanity check
    # Predict 3 for all (neutral) as extra baseline
    neutral_preds = [3] * len(dev)
    results["neutral_3"] = evaluate(neutral_preds, dev_avg, dev_std)

    for name, sc in results.items():
        print(f"  {name:28s} spearman={sc['spearman']:+.4f}  acc@std={sc['accuracy_at_std']:.4f}")

    # Save predictions for majority & per-homonym as jsonl for downstream use
    preds_dir = REPO / "milestone" / "predictions"
    preds_dir.mkdir(exist_ok=True)
    write_predictions(dict(zip(dev_ids, majority_preds)), preds_dir / "majority_dev.jsonl")
    write_predictions(dict(zip(dev_ids, hom_preds)), preds_dir / "per_homonym_mean_dev.jsonl")
    write_predictions(dict(zip(dev_ids, random_preds)), preds_dir / "random_dev.jsonl")

    # Results bar chart
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(results.keys())
    accs = [results[n]["accuracy_at_std"] for n in names]
    sps = [results[n]["spearman"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w/2, accs, w, label="accuracy@std", color="steelblue")
    ax.bar(x + w/2, sps, w, label="Spearman ρ", color="orange")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.axhline(0.7, color="grey", linestyle=":", label="OK solution (acc=0.7)")
    ax.axhline(0.6, color="grey", linestyle="--", label="OK solution (ρ=0.6)")
    ax.set_title("Dev baselines — accuracy@std and Spearman ρ")
    ax.legend(fontsize=8)
    _save(fig, "fig6_baseline_results.png")

    out = REPO / "milestone" / "baseline_scores.json"
    with open(out, "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2)
    print(f"\nWrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
