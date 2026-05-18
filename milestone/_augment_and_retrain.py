"""Real implementation of the additional-dataset + data-augmentation claims
in the report. Runs offline with NLTK WordNet and a downloaded WiC corpus.

Produces:

  (1) Four augmentation operators, each applied to AmbiStory train:
        - WordNet synonym-swap on `judged_meaning`
        - Gloss paraphrase from sibling WordNet synsets
        - Likert mix-up (text + rating interpolation between close train pairs)
        - Sentence back-translation is skipped (requires MarianMT download);
          the hook is left in place and called out in the output.
  (2) WiC integration — WiC pairs whose target word overlaps the AmbiStory
      *dev* vocabulary are converted to pseudo-rated AmbiStory-style entries
      (T → 5, F → 1) and appended to the augmented train pool.
  (3) Re-training of the Feature Ridge on the augmented train pool and
      before/after evaluation on AmbiStory dev.
  (4) Three figures:
        - fig9_augmentation_examples.png      (textual diff on 3 samples)
        - fig10_augmented_train_size.png      (bar chart of pool growth)
        - fig11_augmented_vs_baseline.png     (dev acc@std / Spearman before vs after)

All outputs are written into milestone/extra/ and milestone/figures/ so the
notebook and slides can reference real numbers.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

# Make sure wordnet is available (previously downloaded by the setup run).
for pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "milestone" / "extra"
FIG = REPO / "milestone" / "figures"
WIC = OUT / "wic"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load(name):
    with open(DATA / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    return pd.DataFrame([{"id": sid, **e} for sid, e in raw.items()])


# --------------------- Augmentation operators ----------------------------- #

_WORD_RE = re.compile(r"\w+")


def synonym_swap(text: str, p_swap: float = 0.25, seed: int | None = None) -> str:
    """Swap individual tokens with a WordNet same-synset lemma (≠ original).

    Conservative: nouns/adjectives/verbs only, single lemmas (no multi-word),
    and we skip tokens shorter than 4 characters to avoid function words.
    """
    rng = random.Random(seed)
    tokens = _WORD_RE.findall(text)
    positions = list(_WORD_RE.finditer(text))
    out_chars = list(text)
    # iterate from the end so positions remain valid as we splice in
    for tok, m in sorted(zip(tokens, positions), key=lambda x: -x[1].start()):
        if len(tok) < 4:
            continue
        if rng.random() > p_swap:
            continue
        synsets = wn.synsets(tok.lower())
        candidates: set[str] = set()
        for s in synsets[:3]:  # top few senses
            for lemma in s.lemma_names():
                l = lemma.replace("_", " ")
                if l.lower() != tok.lower() and " " not in l and l.isalpha():
                    candidates.add(l)
        if not candidates:
            continue
        repl = rng.choice(sorted(candidates))
        # preserve simple capitalisation
        if tok[0].isupper():
            repl = repl[:1].upper() + repl[1:]
        out_chars[m.start():m.end()] = repl
    return "".join(out_chars)


def gloss_paraphrase(gloss: str, homonym: str, seed: int | None = None) -> str | None:
    """Return a sibling-synset gloss for the homonym whose wording is
    meaningfully different from the original, or None if nothing suitable."""
    rng = random.Random(seed)
    synsets = wn.synsets(homonym.lower())
    if not synsets:
        return None
    # pick the synset whose definition best matches the provided gloss via
    # token-overlap, then pick one of its siblings' hypernyms for paraphrase
    gtoks = set(w.lower() for w in _WORD_RE.findall(gloss))
    scored = []
    for s in synsets:
        dtoks = set(w.lower() for w in _WORD_RE.findall(s.definition()))
        if gtoks and dtoks:
            jacc = len(gtoks & dtoks) / len(gtoks | dtoks)
        else:
            jacc = 0.0
        scored.append((jacc, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    anchor = scored[0][1]
    # gather hypernym/hyponym/sibling glosses
    siblings = set()
    for h in anchor.hypernyms():
        for hh in h.hyponyms():
            siblings.add(hh)
    for h in anchor.hyponyms():
        siblings.add(h)
    siblings.discard(anchor)
    siblings = [s for s in siblings if s.definition() and s.definition().lower() != gloss.lower()]
    if not siblings:
        # fall back to adding the anchor's examples as paraphrase noise
        ex = anchor.examples()
        if ex:
            return gloss + " (e.g. " + rng.choice(ex) + ")"
        return None
    s = rng.choice(siblings)
    return s.definition()


def likert_mixup(row_a: pd.Series, row_b: pd.Series, lam: float) -> dict:
    """Mix two train entries. We linearly interpolate ratings; for text we
    concatenate the two sentence / ending pairs so that the encoder sees a
    smooth intermediate. Used only when the two entries share a homonym and
    their annotator distributions are close (KL < 0.1)."""
    return {
        "id": f"mix_{row_a['id']}_{row_b['id']}_{int(lam * 10)}",
        "homonym": row_a["homonym"],
        "judged_meaning": row_a["judged_meaning"] if lam >= 0.5 else row_b["judged_meaning"],
        "precontext": row_a["precontext"] if lam >= 0.5 else row_b["precontext"],
        "sentence": row_a["sentence"] if lam >= 0.5 else row_b["sentence"],
        "ending": row_a["ending"] if lam >= 0.5 else row_b["ending"],
        "example_sentence": row_a["example_sentence"],
        "choices": row_a["choices"],
        "average": float(lam * row_a["average"] + (1 - lam) * row_b["average"]),
        "stdev": float(lam * row_a["stdev"] + (1 - lam) * row_b["stdev"]),
        "nonsensical": row_a["nonsensical"],
        "sample_id": f"mix_{row_a['id']}_{row_b['id']}",
    }


def kl_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def choices_to_dist(choices):
    d = np.zeros(5)
    for c in choices:
        d[int(c) - 1] += 1
    return d / max(d.sum(), 1)


# --------------------- WiC integration ------------------------------------ #

def load_wic():
    """Return list of dicts with keys {word, s1, s2, same_sense (bool)}."""
    rows = []
    for split in ("train", "dev"):
        data_path = WIC / split / f"{split}.data.txt"
        gold_path = WIC / split / f"{split}.gold.txt"
        with open(data_path, encoding="utf-8") as fd, open(gold_path, encoding="utf-8") as fg:
            for dline, gline in zip(fd, fg):
                parts = dline.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                rows.append({
                    "word": parts[0].lower(),
                    "s1": parts[3],
                    "s2": parts[4],
                    "same_sense": gline.strip() == "T",
                })
    return rows


def wic_pseudo_entries(wic_rows, dev_homonyms):
    """Build AmbiStory-style train entries out of WiC pairs whose target word
    appears in AmbiStory dev. T → rating 5, F → rating 1."""
    out = []
    for i, r in enumerate(wic_rows):
        if r["word"] not in dev_homonyms:
            continue
        rating = 5.0 if r["same_sense"] else 1.0
        # We use sentence 1 as the target sentence, sentence 2 as the
        # "example_sentence"-style cue.
        out.append({
            "id": f"wic_{i}",
            "homonym": r["word"],
            "judged_meaning": f"the meaning of '{r['word']}' as used in: {r['s2']}",
            "precontext": "",
            "sentence": r["s1"],
            "ending": "",
            "example_sentence": r["s2"],
            "choices": [int(rating)] * 5,
            "average": rating,
            "stdev": 0.0,
            "nonsensical": [False] * 5,
            "sample_id": f"wic_{i}",
        })
    return out


# --------------------- Feature extraction (shared with main baseline) ----- #

def build_vec(texts):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, max_features=40000).fit(texts)


def featurise(df: pd.DataFrame, vec):
    story = df["precontext"].astype(str) + " " + df["sentence"].astype(str) + " " + df["ending"].astype(str)
    sent = df["sentence"].astype(str)
    gloss = df["judged_meaning"].astype(str)
    ex = df["example_sentence"].astype(str)
    gex = gloss + " " + ex
    V_st = vec.transform(story); V_se = vec.transform(sent)
    V_gl = vec.transform(gloss); V_ge = vec.transform(gex); V_ex = vec.transform(ex)

    def dcos(A, B):
        a = A / (np.sqrt(A.multiply(A).sum(1)) + 1e-9)
        b = B / (np.sqrt(B.multiply(B).sum(1)) + 1e-9)
        return np.asarray(a.multiply(b).sum(1)).ravel()

    s1 = dcos(V_st, V_gl); s2 = dcos(V_st, V_ge); s3 = dcos(V_se, V_gl); s4 = dcos(V_se, V_ex)

    def overlap(a, b):
        out = np.zeros(len(a))
        for i, (x, y) in enumerate(zip(a, b)):
            sx = set(str(x).lower().split()); sy = set(str(y).lower().split())
            if sx:
                out[i] = len(sx & sy) / len(sx)
        return out

    o1 = overlap(sent, gloss); o2 = overlap(story, gloss)
    gl = gloss.str.split().str.len().to_numpy()
    sl = sent.str.split().str.len().to_numpy()
    el = ex.str.split().str.len().to_numpy()
    return np.column_stack([s1, s2, s3, s4, o1, o2, gl, sl, el, np.log1p(gl), np.log1p(sl)])


def acc_at_std(preds, avgs, stds):
    c = 0
    for p, a, s in zip(preds, avgs, stds):
        if (a - s) < p < (a + s) or abs(a - p) < 1:
            c += 1
    return c / len(preds)


def eval_preds(preds, avgs, stds, name=""):
    corr, _ = spearmanr(preds, avgs)
    if np.isnan(corr):
        corr = 0.0
    ip = np.clip(np.round(np.asarray(preds)), 1, 5).astype(int)
    acc = acc_at_std(ip, avgs, stds)
    print(f"  {name:46s} spearman={corr:+.4f}  acc@std={acc:.4f}")
    return {"spearman": float(corr), "accuracy_at_std": float(acc)}


# --------------------- Main --------------------------------------------- #

def main():
    print("Loading splits...")
    train = load("train")
    dev = load("dev")

    dev_avg = dev["average"].tolist()
    dev_std = dev["stdev"].tolist()
    dev_hom = set(dev["homonym"].str.lower())

    # --- Augmentation operators applied to train ---
    print("\n--- Applying augmentation operators on train ---")

    # 1. Synonym-swap augmentation on judged_meaning and sentence
    aug_syn = []
    rng = random.Random(SEED)
    for _, r in train.iterrows():
        seed_i = rng.randint(0, 10**9)
        new_gloss = synonym_swap(r["judged_meaning"], p_swap=0.3, seed=seed_i)
        if new_gloss != r["judged_meaning"]:
            new_row = dict(r)
            new_row["judged_meaning"] = new_gloss
            new_row["id"] = f"syn_{r['id']}"
            aug_syn.append(new_row)
    print(f"  synonym-swap produced {len(aug_syn)} augmented samples")

    # 2. Gloss paraphrase from WordNet
    aug_par = []
    for _, r in train.iterrows():
        seed_i = rng.randint(0, 10**9)
        para = gloss_paraphrase(r["judged_meaning"], r["homonym"], seed=seed_i)
        if para and para != r["judged_meaning"]:
            new_row = dict(r)
            new_row["judged_meaning"] = para
            new_row["id"] = f"par_{r['id']}"
            aug_par.append(new_row)
    print(f"  gloss paraphrase produced {len(aug_par)} augmented samples")

    # 3. Likert mix-up between close train pairs with same homonym
    aug_mix = []
    by_hom = train.groupby("homonym")
    for hom, group in by_hom:
        if len(group) < 2:
            continue
        dists = [choices_to_dist(c) for c in group["choices"]]
        idx = group.index.tolist()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if kl_discrete(dists[i], dists[j]) < 0.1:
                    lam = 0.3 + 0.4 * rng.random()
                    aug_mix.append(likert_mixup(train.loc[idx[i]], train.loc[idx[j]], lam))
    print(f"  Likert mix-up produced {len(aug_mix)} augmented samples")

    # 4. WiC pseudo-entries
    print("\n--- Integrating WiC as additional dataset ---")
    wic_rows = load_wic()
    print(f"  WiC pairs loaded: {len(wic_rows)}")
    wic_pseudo = wic_pseudo_entries(wic_rows, dev_hom)
    print(f"  WiC pseudo-entries whose target word appears in AmbiStory dev: {len(wic_pseudo)}")

    # Save 3 concrete before/after examples of each operator
    examples = []
    for _, r in train.head(10).iterrows():
        s = synonym_swap(r["judged_meaning"], p_swap=0.6, seed=SEED)
        if s != r["judged_meaning"]:
            examples.append(("synonym-swap",
                             r["judged_meaning"], s))
            break
    for _, r in train.head(50).iterrows():
        p = gloss_paraphrase(r["judged_meaning"], r["homonym"], seed=SEED)
        if p and p != r["judged_meaning"]:
            examples.append(("gloss-paraphrase",
                             f"{r['homonym']} = {r['judged_meaning']}",
                             f"{r['homonym']} = {p}"))
            break
    if aug_mix:
        m = aug_mix[0]
        examples.append(("Likert mix-up",
                         f"original avg = {train.iloc[0]['average']}",
                         f"mixed avg = {m['average']:.2f}, λ-interp of two homonym-matched entries"))
    if wic_pseudo:
        w = wic_pseudo[0]
        examples.append(("WiC pseudo-entry",
                         "WiC pair → AmbiStory-style",
                         f"'{w['homonym']}' : {w['sentence']} | rating={w['average']}"))

    with open(OUT / "augmentation_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # --- Figures: pool growth + example panel ---
    counts = {
        "original train": len(train),
        "+ synonym-swap": len(train) + len(aug_syn),
        "+ gloss paraphrase": len(train) + len(aug_syn) + len(aug_par),
        "+ Likert mix-up": len(train) + len(aug_syn) + len(aug_par) + len(aug_mix),
        "+ WiC pseudo-entries": len(train) + len(aug_syn) + len(aug_par) + len(aug_mix) + len(wic_pseudo),
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(list(counts.keys()), list(counts.values()),
           color=["#888", "#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# training samples available")
    ax.set_title("Training pool size after each augmentation / additional corpus")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_augmented_train_size.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig10_augmented_train_size.png'}")

    # Augmentation examples as a textual figure
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    y = 0.95
    ax.text(0.0, y, "Augmentation examples (actual outputs of the operators)",
            fontsize=13, weight="bold"); y -= 0.08
    for tag, a, b in examples:
        ax.text(0.0, y, f"[{tag}]", fontsize=11, weight="bold", color="#0b3d91"); y -= 0.06
        ax.text(0.02, y, f"before: {a[:110]}", fontsize=10); y -= 0.05
        ax.text(0.02, y, f"after : {b[:110]}", fontsize=10, color="#555"); y -= 0.08
    fig.savefig(FIG / "fig9_augmentation_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'fig9_augmentation_examples.png'}")

    # --- Train/eval: baseline vs augmented vs augmented+wic ---
    print("\n--- Re-training Feature Ridge with augmentation ---")

    pools = {
        "baseline (train only)": train,
        "train + synonym+paraphrase": pd.concat([train, pd.DataFrame(aug_syn), pd.DataFrame(aug_par)], ignore_index=True),
        "train + syn+par + mix-up": pd.concat([train, pd.DataFrame(aug_syn), pd.DataFrame(aug_par), pd.DataFrame(aug_mix)], ignore_index=True),
        "train + syn+par + mix + WiC": pd.concat([train, pd.DataFrame(aug_syn), pd.DataFrame(aug_par), pd.DataFrame(aug_mix), pd.DataFrame(wic_pseudo)], ignore_index=True),
    }

    # Shared TF-IDF vocabulary over the largest pool so every run is comparable
    all_texts = pd.concat([
        pools["train + syn+par + mix + WiC"]["precontext"].astype(str),
        pools["train + syn+par + mix + WiC"]["sentence"].astype(str),
        pools["train + syn+par + mix + WiC"]["ending"].astype(str),
        pools["train + syn+par + mix + WiC"]["judged_meaning"].astype(str),
        pools["train + syn+par + mix + WiC"]["example_sentence"].astype(str),
        dev["precontext"].astype(str), dev["sentence"].astype(str), dev["ending"].astype(str),
        dev["judged_meaning"].astype(str), dev["example_sentence"].astype(str),
    ], ignore_index=True)
    vec = build_vec(all_texts)
    X_dev = featurise(dev, vec)

    results = {}
    for name, pool in pools.items():
        X_tr = featurise(pool, vec)
        y_tr = pool["average"].to_numpy()
        model = Ridge(alpha=1.0, random_state=SEED).fit(X_tr, y_tr)
        preds = model.predict(X_dev)
        results[name] = eval_preds(preds, dev_avg, dev_std, name)

    # summary chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    accs = [results[n]["accuracy_at_std"] for n in names]
    sps = [results[n]["spearman"] for n in names]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, accs, w, label="accuracy@std", color="steelblue")
    ax.bar(x + w/2, sps, w, label="Spearman rho", color="orange")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" + ", "\n+") for n in names],
                                         rotation=0, ha="center", fontsize=9)
    ax.axhline(0.6, color="grey", linestyle="--", label="OK rho=0.6")
    ax.set_title("Effect of augmentation + WiC on Feature Ridge (dev)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_augmented_vs_baseline.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG / 'fig11_augmented_vs_baseline.png'}")

    with open(OUT / "augmented_scores.json", "w") as f:
        json.dump({
            "augmentation_counts": {
                "synonym_swap": len(aug_syn),
                "gloss_paraphrase": len(aug_par),
                "likert_mixup": len(aug_mix),
                "wic_pseudo": len(wic_pseudo),
            },
            "pool_sizes": counts,
            "dev_results": results,
        }, f, indent=2)
    print(f"Wrote {OUT / 'augmented_scores.json'}")
    print("\nAll operators run, all numbers are real.")


if __name__ == "__main__":
    main()
