"""Data loading and split utilities for SemEval-2026 Task 5 (AmbiStory).

The official archive ships ``data/{train,dev,test}.json`` as a JSON object that
maps a *string* sample id to an entry with 11 fields::

    homonym, judged_meaning, precontext, sentence, ending, example_sentence,
    choices (5 annotator Likert votes), average, stdev, nonsensical (5 bools),
    sample_id

``test.json`` has the label fields redacted to the string ``"(???)"`` — only
predictions can be produced for it.

Two facts drive everything in this file:

1.  The published splits are *near homonym-disjoint*: ``dev`` shares **0**
    homonyms with ``train`` and ``test`` shares **1**.  A model selected on dev
    therefore would not transfer to test, so we never tune on dev.  Instead we
    carve a **homonym-disjoint hold-out** out of train (``make_holdout``) that
    reproduces the dev/test regime and use *that* for early stopping, calibration
    and ensemble-weight search.

2.  The official scorer (``scoring.py``) wants gold as JSONL lines
    ``{"id": <str>, "label": [v1..v5]}`` whereas the data file stores the five
    votes under ``choices``.  ``write_solution_jsonl`` bridges the two so we can
    self-score with the *real* script rather than a re-implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root = two levels up from this file (final/src/data.py -> repo).
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

SEED = 42
#: ids in the data files are strings ("0", "1", ...).  Keep them as strings
#: everywhere — the official format_check compares ``str(id)``.
ID_COL = "id"

#: Columns that are redacted in test.json.
_REDACTED = "(???)"


def load_split(name: str) -> pd.DataFrame:
    """Load ``data/<name>.json`` into a DataFrame, preserving string ids.

    Label columns (``average``, ``stdev``, ``choices``, ``nonsensical``) are
    coerced to numeric/native types for train and dev; for test they stay as the
    redaction sentinel and are simply not used.
    """
    with open(DATA / f"{name}.json", encoding="utf-8") as fh:
        raw = json.load(fh)

    rows = []
    for sid, entry in raw.items():
        row = {ID_COL: str(sid), **entry, "split": name}
        rows.append(row)
    df = pd.DataFrame(rows)

    has_labels = _REDACTED not in set(df["average"].astype(str))
    if has_labels:
        df["average"] = df["average"].astype(float)
        df["stdev"] = df["stdev"].astype(float)
        # ``choices`` is a list[int]; keep it as a python list per row.
        df["choices"] = df["choices"].apply(lambda c: [int(x) for x in c])
    df.attrs["has_labels"] = has_labels
    return df


def build_story(df: pd.DataFrame) -> pd.Series:
    """Full 5-sentence narrative: precontext + target sentence + ending."""
    return (
        df["precontext"].astype(str).str.strip()
        + " "
        + df["sentence"].astype(str).str.strip()
        + " "
        + df["ending"].astype(str).str.strip()
    ).str.strip()


def build_pair_text(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Gloss-informed encoder input (Component B / C).

    Segment A is the *target sentence* (where the homonym actually occurs);
    segment B places the candidate gloss right next to it so cross-attention is
    anchored on the lexical item, then appends the example and the surrounding
    story.  Recipe adapted from Blevins & Zettlemoyer (ACL 2020).
    """
    seg_a = df["sentence"].astype(str).str.strip()
    seg_b = (
        df["homonym"].astype(str).str.strip()
        + ": "
        + df["judged_meaning"].astype(str).str.strip()
        + " [example] "
        + df["example_sentence"].astype(str).str.strip()
        + " [story] "
        + df["precontext"].astype(str).str.strip()
        + " "
        + df["ending"].astype(str).str.strip()
    )
    return seg_a, seg_b


def empirical_distribution(choices: list[int]) -> np.ndarray:
    """5-vote list -> probability vector over Likert values {1..5}."""
    counts = np.zeros(5, dtype=np.float64)
    for c in choices:
        counts[int(c) - 1] += 1.0
    return counts / counts.sum()


def make_holdout(
    train: pd.DataFrame, frac: float = 0.12, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Homonym-disjoint hold-out of ``train`` that mirrors the dev/test regime.

    Whole homonyms (not random rows) are moved to the hold-out, so no homonym
    appears in both folds — exactly the situation the model faces on dev/test.
    Returns boolean masks ``(fit_mask, holdout_mask)`` aligned to ``train``.

    Homonyms are shuffled with a fixed RNG and accumulated into the hold-out
    until it reaches ``frac`` of the rows; this is deterministic for a seed.
    """
    rng = np.random.default_rng(seed)
    homonyms = train["homonym"].astype(str).to_numpy()
    uniq = np.array(sorted(set(homonyms)))
    rng.shuffle(uniq)

    target = int(round(frac * len(train)))
    holdout_homs: set[str] = set()
    n_held = 0
    for h in uniq:
        if n_held >= target:
            break
        holdout_homs.add(h)
        n_held += int((homonyms == h).sum())

    holdout_mask = np.array([h in holdout_homs for h in homonyms])
    fit_mask = ~holdout_mask
    assert holdout_mask.any() and fit_mask.any(), "degenerate hold-out split"
    # sanity: the two folds must not share any homonym
    assert not (set(homonyms[fit_mask]) & set(homonyms[holdout_mask])), (
        "hold-out is not homonym-disjoint"
    )
    return fit_mask, holdout_mask


def write_solution_jsonl(df: pd.DataFrame, out_path: Path) -> Path:
    """Write gold in the layout the official ``scoring.py`` expects.

    ``scoring.py`` reads ``line["label"]`` (the five votes) and computes the
    average/stdev itself, so we emit ``{"id", "label": choices}``.  Only valid
    for splits that still have labels (train / dev).
    """
    assert df.attrs.get("has_labels", True), "cannot write solution for redacted split"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({"id": str(r[ID_COL]), "label": list(r["choices"])}) + "\n")
    return out_path


def write_predictions_jsonl(
    ids: list[str], preds, out_path: Path, *, as_int: bool = False
) -> Path:
    """Write predictions JSONL in submission format.

    ``scoring.py`` consumes the raw value for both metrics, so by default we
    keep the calibrated *continuous* prediction (clipped to [1,5]) which is the
    single largest Spearman lever.  ``as_int`` emits the rounded fallback used
    for the strict-format CodaBench variant and the continuous-vs-int ablation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(preds, dtype=np.float64), 1.0, 5.0)
    with open(out_path, "w", encoding="utf-8") as fh:
        for sid, p in zip(ids, arr):
            value = int(round(float(p))) if as_int else round(float(p), 4)
            fh.write(json.dumps({"id": str(sid), "prediction": value}) + "\n")
    return out_path
