"""Extra milestone contribution — light, offline, genuinely-signal baselines
that go beyond the constant predictors in the repo.

Produces real numbers that we cite in the report and slides:

  (1) TF-IDF cosine similarity between the *full story* and the candidate
      gloss + example_sentence, calibrated linearly on train.
  (2) A feature-based Ridge regressor over a handful of lexical similarity
      features (cosine of the story vs. gloss at different spans, length
      features, and unigram-overlap counts).
  (3) The same Ridge scores thresholded via a 4-threshold ordinal grid
      search tuned on a train hold-out (addresses the ordinal nature of the
      label, per our report's ordinal-regression plan).
  (4) Likert-distribution conditional plots motivating the distribution head.

Pure numpy/pandas/sklearn/scipy — no network, no GPU.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "milestone" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
PREDS = REPO / "milestone" / "predictions"
FIG = REPO / "milestone" / "figures"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load(name):
    with open(DATA / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    return pd.DataFrame([{"id": sid, **e} for sid, e in raw.items()])


def acc_at_std(preds, avgs, stds):
    correct = 0
    for p, a, s in zip(preds, avgs, stds):
        if (a - s) < p < (a + s) or abs(a - p) < 1:
            correct += 1
    return correct / len(preds)


def evaluate(preds, avgs, stds, name=""):
    corr, _ = spearmanr(preds, avgs)
    if np.isnan(corr):
        corr = 0.0
    int_preds = np.clip(np.round(np.asarray(preds)), 1, 5).astype(int)
    acc = acc_at_std(int_preds, avgs, stds)
    print(f"  {name:42s} spearman={corr:+.4f}  acc@std={acc:.4f}")
    return {"spearman": float(corr), "accuracy_at_std": float(acc),
            "int_predictions": int_preds.tolist(),
            "cont_predictions": [float(x) for x in np.asarray(preds).tolist()]}


def write_preds_jsonl(ids, preds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, p in zip(ids, preds):
            f.write(json.dumps({"id": str(i), "prediction": int(p)}) + "\n")


def build_features(df: pd.DataFrame, vec: TfidfVectorizer):
    """Return a numeric feature matrix with similarity + length features."""
    story = (df["precontext"].astype(str) + " "
             + df["sentence"].astype(str) + " "
             + df["ending"].astype(str))
    sentence = df["sentence"].astype(str)
    gloss = df["judged_meaning"].astype(str)
    example = df["example_sentence"].astype(str)
    gloss_plus_ex = gloss + " " + example

    V_story = vec.transform(story)
    V_sent = vec.transform(sentence)
    V_gloss = vec.transform(gloss)
    V_gloss_ex = vec.transform(gloss_plus_ex)
    V_example = vec.transform(example)

    def diag_cos(A, B):
        # pairwise cosine between row_i(A) and row_i(B)
        a_norm = A / (np.sqrt(A.multiply(A).sum(1)) + 1e-9)
        b_norm = B / (np.sqrt(B.multiply(B).sum(1)) + 1e-9)
        return np.asarray(a_norm.multiply(b_norm).sum(1)).ravel()

    sim_story_gloss = diag_cos(V_story, V_gloss)
    sim_story_glossex = diag_cos(V_story, V_gloss_ex)
    sim_sent_gloss = diag_cos(V_sent, V_gloss)
    sim_sent_example = diag_cos(V_sent, V_example)

    gloss_len = gloss.str.split().str.len().to_numpy()
    sent_len = sentence.str.split().str.len().to_numpy()
    example_len = example.str.split().str.len().to_numpy()

    # word-level overlap between sentence tokens and gloss tokens
    def overlap_ratio(a, b):
        out = np.zeros(len(a))
        for i, (x, y) in enumerate(zip(a, b)):
            sx = set(str(x).lower().split())
            sy = set(str(y).lower().split())
            if sx:
                out[i] = len(sx & sy) / len(sx)
        return out

    overlap_sent_gloss = overlap_ratio(sentence, gloss)
    overlap_story_gloss = overlap_ratio(story, gloss)

    feats = np.column_stack([
        sim_story_gloss,
        sim_story_glossex,
        sim_sent_gloss,
        sim_sent_example,
        overlap_sent_gloss,
        overlap_story_gloss,
        gloss_len,
        sent_len,
        example_len,
        np.log1p(gloss_len),
        np.log1p(sent_len),
    ])
    return feats, sim_story_glossex  # return primary similarity for baseline (1)


def main():
    print("Loading splits...")
    train = load("train")
    dev = load("dev")
    dev_avg = dev["average"].tolist()
    dev_std = dev["stdev"].tolist()
    y_train = train["average"].to_numpy()

    # Fit one shared TF-IDF on the full corpus (train + dev texts) so both
    # splits use the same vocabulary. This does NOT leak labels.
    all_text = pd.concat([
        train["precontext"], train["sentence"], train["ending"],
        train["judged_meaning"], train["example_sentence"],
        dev["precontext"], dev["sentence"], dev["ending"],
        dev["judged_meaning"], dev["example_sentence"],
    ], ignore_index=True).astype(str)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=40000)
    vec.fit(all_text)

    X_train, sim_tr = build_features(train, vec)
    X_dev, sim_dev = build_features(dev, vec)

    results: dict[str, dict] = {}

    print("\n--- (1) TF-IDF cosine (story vs gloss+example), calibrated ---")
    calib = np.polyfit(sim_tr, y_train, 1)
    print(f"  linear calibration: rating = {calib[0]:+.3f} * sim + {calib[1]:+.3f}")
    cos_pred = np.polyval(calib, sim_dev)
    results["tfidf_cosine"] = evaluate(cos_pred, dev_avg, dev_std,
                                       "TF-IDF cosine story vs gloss+ex")

    print("\n--- (2) Feature-based Ridge regression ---")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train, y_train)
    cont_preds = ridge.predict(X_dev)
    results["feature_ridge"] = evaluate(cont_preds, dev_avg, dev_std,
                                        "Feature Ridge (9-dim)")
    # coefficients for insight
    coef_names = ["sim_story_gloss", "sim_story_glossex", "sim_sent_gloss",
                  "sim_sent_example", "overlap_sent_gloss", "overlap_story_gloss",
                  "gloss_len", "sent_len", "example_len", "log_gloss_len", "log_sent_len"]
    print("  ridge coefficients:")
    for n, c in zip(coef_names, ridge.coef_):
        print(f"    {n:22s} {c:+.4f}")

    write_preds_jsonl(dev["id"].tolist(),
                      results["feature_ridge"]["int_predictions"],
                      PREDS / "feature_ridge_dev.jsonl")

    print("\n--- (3) Ordinal threshold tuning on ridge continuous output ---")
    # held-out 10% of train for threshold selection
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(train))
    rng.shuffle(idx)
    cut = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    ridge_tr = Ridge(alpha=1.0, random_state=SEED)
    ridge_tr.fit(X_train[tr_idx], y_train[tr_idx])
    va_scores = ridge_tr.predict(X_train[va_idx])
    va_avg = y_train[va_idx]
    va_std = train["stdev"].to_numpy()[va_idx]
    best = None
    # coarse then refined — keep it reasonable
    for t1 in np.linspace(1.6, 2.6, 11):
        for t2 in np.linspace(t1 + 0.1, 3.3, 9):
            for t3 in np.linspace(t2 + 0.1, 4.0, 9):
                for t4 in np.linspace(t3 + 0.1, 4.8, 9):
                    buckets = np.digitize(va_scores, [t1, t2, t3, t4]) + 1
                    a = acc_at_std(buckets, va_avg, va_std)
                    if best is None or a > best[0]:
                        best = (a, (float(t1), float(t2), float(t3), float(t4)))
    thr = best[1]
    print(f"  best thresholds: {['%.2f'%t for t in thr]}  (train-holdout acc@std={best[0]:.4f})")
    dev_ord = np.digitize(cont_preds, list(thr)) + 1
    results["feature_ridge_ordinal"] = evaluate(
        dev_ord, dev_avg, dev_std, "Feature Ridge + ordinal thresholds")
    write_preds_jsonl(dev["id"].tolist(), dev_ord,
                      PREDS / "feature_ridge_ordinal_dev.jsonl")

    print("\n--- (4) Likert distribution analysis ---")
    buckets = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.001)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.2
    xs = np.arange(1, 6)
    for i, (lo, hi) in enumerate(buckets):
        mask = (train["average"] >= lo) & (train["average"] < hi)
        sub = train[mask]
        counts = np.zeros(5)
        for row in sub["choices"]:
            for c in row:
                counts[int(c) - 1] += 1
        if counts.sum() > 0:
            counts = counts / counts.sum()
        ax.bar(xs + (i - 1.5) * width, counts, width,
               label=f"avg in [{lo:.1f},{hi:.1f}) n={int(mask.sum())}")
    ax.set_xlabel("Likert rating value"); ax.set_ylabel("P(choice | avg-bucket)")
    ax.set_title("Train: Likert distribution conditional on average rating")
    ax.legend(fontsize=8); ax.set_xticks(xs)
    fig.savefig(FIG / "fig7_likert_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / 'fig7_likert_distribution.png'}")

    print("\n--- (5) Combined summary bar chart ---")
    summary_rows = [
        ("majority (=4)",                       0.5697, 0.0000),
        ("random [1,5]",                        0.4371, 0.0334),
        ("mean-of-train (=3)",                  0.5272, 0.0000),
        ("TF-IDF cosine (calibrated)",
            results["tfidf_cosine"]["accuracy_at_std"],
            results["tfidf_cosine"]["spearman"]),
        ("Feature Ridge",
            results["feature_ridge"]["accuracy_at_std"],
            results["feature_ridge"]["spearman"]),
        ("Feature Ridge + ordinal",
            results["feature_ridge_ordinal"]["accuracy_at_std"],
            results["feature_ridge_ordinal"]["spearman"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r[0] for r in summary_rows]
    accs = [r[1] for r in summary_rows]
    sps = [r[2] for r in summary_rows]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, accs, w, label="accuracy@std", color="steelblue")
    ax.bar(x + w/2, sps, w, label="Spearman rho", color="orange")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=22, ha="right")
    ax.axhline(0.7, color="grey", linestyle=":", label="OK acc=0.7")
    ax.axhline(0.6, color="grey", linestyle="--", label="OK rho=0.6")
    ax.set_title("Dev baselines + extra milestone contribution")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig8_extra_baselines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / 'fig8_extra_baselines.png'}")

    summary = {
        "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("int_predictions", "cont_predictions")}
                    for k, v in results.items()},
        "ridge_coefficients": {n: float(c) for n, c in zip(coef_names, ridge.coef_)},
        "ordinal_thresholds": list(thr),
    }
    with open(OUT / "extra_scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {(OUT / 'extra_scores.json')}")
    print("\nFinal summary:")
    for n, a, s in summary_rows:
        print(f"  {n:40s} acc@std={a:.4f}  rho={s:+.4f}")


if __name__ == "__main__":
    main()
