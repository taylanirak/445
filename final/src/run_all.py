"""End-to-end orchestration for the SemEval-2026 Task 5 final system.

Pipeline
--------
1.  Load splits, build the homonym-disjoint train hold-out and the gloss
    informed text fields; write ``_solution_dev.jsonl`` for official scoring.
2.  Reproducible lexical baselines (TF-IDF cosine, feature Ridge) — carried
    forward from the milestone so the report can show the journey / SOTA bar.
3.  Component A (zero-shot NLI), B (fine-tuned regressor), C (novel Likert
    distribution head).  Each emits continuous hold-out/dev/test predictions.
4.  Ensemble: hold-out weight search + isotonic calibration.
5.  Score every system with the *official* ``scoring.py``; run the ablations
    (continuous-vs-int, calibration, drop-one, CORN, uncertainty-gate).
6.  Write all prediction files and a single ``results.json`` that the figures,
    notebook and report all read (one source of truth).

Runtime control
---------------
* ``FAST=1``  — 1 seed, ``roberta-base``, 2 epochs (~45 min GPU / quick CPU).
* No GPU      — auto-fallback to ``roberta-base``, 1 epoch, train subsample,
  with a printed WARNING banner (numbers will be weaker; the canonical run is
  GPU + ``deberta-v3-large``).
* ``SKIP_HEAVY=1`` — baselines + ensemble scaffold only (no model downloads);
  used for fast structural CI of the harness.

All randomness is seeded (SEED=42); HF downloads are cached under
``final/.hf_cache``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
FINAL = THIS.parents[1]
REPO = THIS.parents[2]
sys.path.insert(0, str(FINAL))  # allow ``import src.*`` when run as a script

os.environ.setdefault("HF_HOME", str(FINAL / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src import data as D  # noqa: E402
from src import metrics as M  # noqa: E402

# --------------------------------------------------------------------------- #
# Component-level cache — a heavy CPU run can be OOM-killed mid-pipeline; once a
# component (A/B/C) has produced its hold-out/dev/test predictions we persist
# them, so a re-run skips the expensive retrain and only redoes what's missing.
# The cache key encodes the full config, so a different (e.g. GPU) setup never
# reuses CPU artifacts.
# --------------------------------------------------------------------------- #
import gc  # noqa: E402
import hashlib  # noqa: E402

CACHE = FINAL / ".cache"

# Bump this whenever the *training / model / loss code* changes (not just
# config). _run_sig hashes config only; without this constant a code change with
# an unchanged config would silently reload the stale <sig>_{A,B,C}.npz and the
# new code would never run. Folding CODE_VERSION into the signature guarantees a
# re-run actually exercises the updated components.
CODE_VERSION = "2026-05-18-a"


def _run_sig(rt: dict, nli_model: str, n_tmpl: int) -> str:
    raw = (f"{CODE_VERSION}|{rt['backbone']}|{rt['max_len']}|{rt['epochs']}|"
           f"{rt['seeds']}|{rt['grad_accum']}|{rt['subsample']}|{nli_model}|"
           f"{n_tmpl}")
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _cache_load(sig: str, name: str):
    f = CACHE / f"{sig}_{name}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    print(f"[cache] loaded component {name} from {f.name}")
    return out


def _cache_save(sig: str, name: str, **arrays) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{sig}_{name}.npz", **arrays)
    print(f"[cache] saved component {name}")


def _free():
    """Release memory between heavy components (helps the 8 GB CPU path)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

SEED = 42
PRED_DIR = FINAL / "predictions"
RESULTS = FINAL / "results.json"
TARGETS = {"OK": {"accuracy_at_std": 0.70, "spearman": 0.60},
           "Good": {"accuracy_at_std": 0.80, "spearman": 0.70}}


# --------------------------------------------------------------------------- #
# Runtime configuration
# --------------------------------------------------------------------------- #
def _resolve_runtime():
    fast = os.environ.get("FAST", "0") == "1"
    skip_heavy = os.environ.get("SKIP_HEAVY", "0") == "1"
    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        pass

    max_len, batch, grad_accum = 256, 16, 1
    if has_gpu and not fast:
        backbone, epochs, seeds, sub = "microsoft/deberta-v3-large", 4, (13, 42, 123), None
        mode = "GPU / deberta-v3-large (canonical)"
    elif has_gpu and fast:
        backbone, epochs, seeds, sub = "roberta-base", 2, (42,), None
        mode = "GPU / FAST (roberta-base)"
    else:
        # Strong CPU run: keep the memory-safe distilroberta-base / seq128 /
        # batch8 envelope (peak RSS ~1.7 GB, fits 8 GB) but spend the extra
        # budget on full training data + more epochs + 2 NLI templates — these
        # raise quality WITHOUT raising peak RAM, only wall-time. The canonical
        # GPU path above is unaffected.
        backbone, epochs, seeds, sub = "roberta-base", 4, (42,), None
        max_len, batch, grad_accum = 128, 4, 4  # effective batch 16, low peak RAM
        mode = "CPU (roberta-base, seq128, bs4×ga4, full data, 4 epochs) — weaker than canonical GPU"
        print("=" * 72)
        print("WARNING: no CUDA GPU detected. Running the lightweight CPU")
        print("         fallback (distilroberta-base). Numbers are below the")
        print("         canonical GPU run — re-run on a GPU to reproduce them.")
        print("=" * 72)
    return dict(fast=fast, skip_heavy=skip_heavy, has_gpu=has_gpu,
                backbone=backbone, epochs=epochs, seeds=seeds,
                subsample=sub, max_len=max_len, batch=batch,
                grad_accum=grad_accum, mode=mode)


# --------------------------------------------------------------------------- #
# Lexical baselines (carried forward from the milestone)
# --------------------------------------------------------------------------- #
def lexical_baselines(train, dev, sol_path):
    """Constant + TF-IDF cosine + feature-Ridge baselines, officially scored."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    out = {}

    def off(preds, tag):
        D.write_predictions_jsonl(
            dev["id"].tolist(), preds, PRED_DIR / f"_baseline_{tag}.jsonl"
        )
        s = M.official_score(
            sol_path, PRED_DIR / f"_baseline_{tag}.jsonl",
            FINAL / "_scratch_scores.json",
        )
        return {"accuracy_at_std": s["accuracy"], "spearman": s["spearman"]}

    out["majority_4"] = off([4] * len(dev), "majority")
    out["mean_of_train"] = off(
        [round(float(train["average"].mean()))] * len(dev), "mean"
    )

    story_tr = D.build_story(train)
    story_dv = D.build_story(dev)
    gloss_tr = train["judged_meaning"].astype(str) + " " + train["example_sentence"].astype(str)
    gloss_dv = dev["judged_meaning"].astype(str) + " " + dev["example_sentence"].astype(str)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=40000)
    vec.fit(list(story_tr) + list(gloss_tr) + list(story_dv) + list(gloss_dv))

    def cos(a, b):
        A, B = vec.transform(a), vec.transform(b)
        an = A / (np.sqrt(A.multiply(A).sum(1)) + 1e-9)
        bn = B / (np.sqrt(B.multiply(B).sum(1)) + 1e-9)
        return np.asarray(an.multiply(bn).sum(1)).ravel()

    s_tr, s_dv = cos(story_tr, gloss_tr), cos(story_dv, gloss_dv)
    cal = np.polyfit(s_tr, train["average"].to_numpy(), 1)
    out["tfidf_cosine"] = off(np.clip(np.polyval(cal, s_dv), 1, 5), "tfidf")

    X_tr = np.column_stack([s_tr, gloss_tr.str.split().str.len(),
                            train["sentence"].str.split().str.len()])
    X_dv = np.column_stack([s_dv, gloss_dv.str.split().str.len(),
                            dev["sentence"].str.split().str.len()])
    rg = Ridge(alpha=1.0, random_state=SEED).fit(X_tr, train["average"].to_numpy())
    out["feature_ridge"] = off(np.clip(rg.predict(X_dv), 1, 5), "ridge")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    rt = _resolve_runtime()
    print(f"[run_all] mode = {rt['mode']}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    train = D.load_split("train")
    dev = D.load_split("dev")
    test = D.load_split("test")
    fit_mask, hold_mask = D.make_holdout(train, frac=0.12, seed=SEED)

    sol_path = D.write_solution_jsonl(dev, FINAL / "_solution_dev.jsonl")

    hom = lambda df: set(df["homonym"].astype(str))
    meta = {
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "holdout": {"fit": int(fit_mask.sum()), "holdout": int(hold_mask.sum()),
                    "holdout_homonyms": int(len(hom(train[hold_mask])))},
        "homonym_overlap": {
            "dev_in_train": len(hom(dev) & hom(train)),
            "test_in_train": len(hom(test) & hom(train)),
        },
        "label_stats": {
            "mean_avg": float(train["average"].mean()),
            "mean_stdev": float(train["stdev"].mean()),
            "frac_high_disagreement": float((train["stdev"] >= 1.2).mean()),
        },
        "runtime_mode": rt["mode"],
    }
    print(f"[run_all] data: {meta['sizes']} | hold-out {meta['holdout']} | "
          f"overlap {meta['homonym_overlap']}")

    results = {"meta": meta, "targets": TARGETS, "baselines": {},
               "components": {}, "ensemble": {}, "ablations": {},
               "classification": {}}

    print("\n[run_all] --- lexical baselines (milestone carry-forward) ---")
    results["baselines"] = lexical_baselines(train, dev, sol_path)
    for k, v in results["baselines"].items():
        print(f"  {k:16s} acc@std={v['accuracy_at_std']:.4f} "
              f"rho={v['spearman']:+.4f}")

    if rt["skip_heavy"]:
        print("\n[run_all] SKIP_HEAVY=1 — stopping after baselines.")
        RESULTS.write_text(json.dumps(results, indent=2))
        return

    # ---- shared text fields for all neural components --------------------- #
    fit = train[fit_mask].reset_index(drop=True)
    hold = train[hold_mask].reset_index(drop=True)
    if rt["subsample"] and len(fit) > rt["subsample"]:
        fit = fit.sample(rt["subsample"], random_state=SEED).reset_index(drop=True)
        print(f"[run_all] CPU fallback: subsampled fit fold to {len(fit)} rows")

    fa, fb = D.build_pair_text(fit)
    ha, hb = D.build_pair_text(hold)
    da, db = D.build_pair_text(dev)
    ta, tb = D.build_pair_text(test)
    bundle = {
        "fit_a": fa, "fit_b": fb, "fit_avg": fit["average"].to_numpy(),
        "fit_choices": list(fit["choices"]),
        "hold_a": ha, "hold_b": hb, "hold_avg": hold["average"].to_numpy(),
        "dev_a": da, "dev_b": db, "test_a": ta, "test_b": tb,
    }
    hold_avg = hold["average"].to_numpy()
    dev_avg = dev["average"].to_numpy()
    dev_std = dev["stdev"].to_numpy()
    ids_dev = dev["id"].tolist()
    ids_test = test["id"].tolist()

    holdout_preds, dev_preds, test_preds = {}, {}, {}

    # ---- Component A : zero-shot NLI -------------------------------------- #
    print("\n[run_all] --- Component A: zero-shot NLI ---")
    from src.nli_scorer import Calibrator, FALLBACK_MODEL, PRIMARY_MODEL, nli_raw_scores

    nli_model = PRIMARY_MODEL if rt["has_gpu"] else FALLBACK_MODEL
    # Canonical GPU: 2 templates, full fit fold for calibration. CPU: 1 template
    # + larger batch + a capped calibration subsample (a few hundred points are
    # plenty for an isotonic/linear fit) so the run stays tractable.
    n_tmpl = 2  # 2 paraphrased hypotheses on both paths (quality > a bit of time)
    nli_bs = 16 if rt["has_gpu"] else 64
    calib_df = (fit if rt["has_gpu"]
                else fit.sample(min(len(fit), 1000), random_state=SEED))
    sig = _run_sig(rt, nli_model, n_tmpl)
    print(f"[run_all] cache signature = {sig}")
    try:
        cA = _cache_load(sig, "A")
        if cA is None:
            nli = lambda df: nli_raw_scores(
                list(D.build_story(df)), list(df["homonym"]),
                list(df["judged_meaning"]), model_name=nli_model,
                batch_size=nli_bs, n_templates=n_tmpl)
            raw_fit = nli(calib_df)
            raw_hold = nli(hold)
            raw_dev = nli(dev)
            raw_test = nli(test)
            calib = Calibrator("isotonic").fit(
                raw_fit, calib_df["average"].to_numpy())
            calib_lin = Calibrator("linear").fit(
                raw_fit, calib_df["average"].to_numpy())
            A_hold = calib.predict(raw_hold)
            A_dev = calib.predict(raw_dev)
            A_test = calib.predict(raw_test)
            A_lin_dev = calib_lin.predict(raw_dev)
            _cache_save(sig, "A", hold=A_hold, dev=A_dev, test=A_test,
                        lin_dev=A_lin_dev)
            _free()
        else:
            A_hold, A_dev, A_test, A_lin_dev = (
                cA["hold"], cA["dev"], cA["test"], cA["lin_dev"])
        holdout_preds["A"], dev_preds["A"], test_preds["A"] = A_hold, A_dev, A_test
        results["components"]["A_nli"] = {
            "model": nli_model,
            "dev": _score(ids_dev, dev_preds["A"], sol_path),
            "holdout_spearman": M.spearman(holdout_preds["A"], hold_avg),
            "calibration_linear_dev_spearman": M.spearman(A_lin_dev, dev_avg),
        }
        results["ablations"]["nli_calibration"] = {
            "isotonic": results["components"]["A_nli"]["dev"]["spearman"],
            "linear": results["components"]["A_nli"]["calibration_linear_dev_spearman"],
        }
    except Exception as e:  # keep the pipeline alive; report the failure
        print(f"[run_all] Component A failed: {e!r}")
        results["components"]["A_nli"] = {"error": repr(e)}

    # ---- Component B : fine-tuned regressor ------------------------------ #
    print("\n[run_all] --- Component B: fine-tuned gloss-informed regressor ---")
    from src.encoder_reg import TrainConfig, train_regressor

    cfgB = TrainConfig(model_name=rt["backbone"], epochs=rt["epochs"],
                       seeds=rt["seeds"], fp16=rt["has_gpu"],
                       max_length=rt["max_len"], batch_size=rt["batch"],
                       grad_accum=rt["grad_accum"],
                       lr=1e-5 if "deberta-v3-large" in rt["backbone"] else 2e-5)
    cB = _cache_load(sig, "B")
    if cB is None:
        rB = train_regressor(cfgB, bundle, "EncReg")
        _cache_save(sig, "B", hold=rB["holdout"], dev=rB["dev"],
                    test=rB["test"], rho=np.array(rB["best_holdout_rho"]),
                    curves=np.array(rB["holdout_curves"], dtype=object))
        _free()
        b_hold, b_dev, b_test = rB["holdout"], rB["dev"], rB["test"]
        b_rho, b_curves = rB["best_holdout_rho"], rB["holdout_curves"]
    else:
        b_hold, b_dev, b_test = cB["hold"], cB["dev"], cB["test"]
        b_rho = float(cB["rho"]); b_curves = cB["curves"].tolist()
    holdout_preds["B"] = b_hold
    dev_preds["B"], test_preds["B"] = b_dev, b_test
    results["components"]["B_encoder"] = {
        "model": rt["backbone"], "seeds": list(rt["seeds"]),
        "best_holdout_rho": b_rho,
        "holdout_curves": b_curves,
        "dev": _score(ids_dev, dev_preds["B"], sol_path),
    }

    # ---- Component C : novel Likert-distribution head -------------------- #
    print("\n[run_all] --- Component C: novel Likert-distribution head ---")
    from src.likert_head import ObjectiveConfig, train_distribution_head

    cC = _cache_load(sig, "C")
    if cC is None:
        rC = train_distribution_head(cfgB, bundle,
                                     ObjectiveConfig("distribution"), "LikertHead")
        _cache_save(sig, "C", hold=rC["holdout"], dev=rC["dev"],
                    test=rC["test"], dev_var=rC["dev_var"],
                    rho=np.array(rC["best_holdout_rho"]),
                    curves=np.array(rC["holdout_curves"], dtype=object))
        _free()
        c_hold, c_dev, c_test, dev_var_C = (
            rC["holdout"], rC["dev"], rC["test"], rC["dev_var"])
        c_rho, c_curves = rC["best_holdout_rho"], rC["holdout_curves"]
    else:
        c_hold, c_dev, c_test, dev_var_C = (
            cC["hold"], cC["dev"], cC["test"], cC["dev_var"])
        c_rho = float(cC["rho"]); c_curves = cC["curves"].tolist()
    holdout_preds["C"] = c_hold
    dev_preds["C"], test_preds["C"] = c_dev, c_test
    results["components"]["C_likert"] = {
        "model": rt["backbone"], "objective": "distribution",
        "best_holdout_rho": c_rho,
        "holdout_curves": c_curves,
        "dev": _score(ids_dev, dev_preds["C"], sol_path),
    }

    _free()
    # Head-objective ablation. distribution vs MSE are always available from
    # Components C and B. The CORN ordinal head is a *third* full backbone
    # training; on the low-RAM CPU path that extra model OOM-kills the process,
    # so CORN runs only on GPU or when explicitly requested (RUN_CORN=1) or
    # already cached. Its absence degrades to "n/a" everywhere — honest, and the
    # headline distribution-vs-MSE comparison is unaffected.
    head_obj = {
        "distribution_KL": results["components"]["C_likert"]["dev"]["spearman"],
        "mse_regressor": results["components"]["B_encoder"]["dev"]["spearman"],
    }
    run_corn = rt["has_gpu"] or os.environ.get("RUN_CORN") == "1"
    cCorn = _cache_load(sig, "CORN")
    if cCorn is not None:
        head_obj["corn_ordinal"] = _score(
            ids_dev, cCorn["dev"], sol_path)["spearman"]
    elif run_corn:
        try:
            cfgAbl = TrainConfig(
                model_name=rt["backbone"], epochs=max(1, rt["epochs"] - 1),
                seeds=(42,), fp16=rt["has_gpu"], max_length=rt["max_len"],
                batch_size=rt["batch"], grad_accum=rt["grad_accum"],
                lr=1e-5 if "deberta-v3-large" in rt["backbone"] else 2e-5)
            rCorn = train_distribution_head(cfgAbl, bundle,
                                            ObjectiveConfig("corn"), "CORN")
            _cache_save(sig, "CORN", dev=rCorn["dev"])
            _free()
            head_obj["corn_ordinal"] = _score(
                ids_dev, rCorn["dev"], sol_path)["spearman"]
        except Exception as e:
            print(f"[run_all] CORN ablation skipped: {e!r}")
    else:
        print("[run_all] CORN ablation skipped on low-RAM CPU path "
              "(set RUN_CORN=1 to force).")
    results["ablations"]["head_objective"] = head_obj

    # ---- Ensemble -------------------------------------------------------- #
    print("\n[run_all] --- Ensemble (hold-out weight search + isotonic) ---")
    from src.ensemble import Ensemble, uncertainty_gate, write_all_predictions

    if not all(k in holdout_preds for k in "ABC"):
        # If A failed, fall back to a 2-component ensemble by zero-filling A.
        for miss in set("ABC") - set(holdout_preds):
            holdout_preds[miss] = np.full(len(hold_avg), float(np.mean(hold_avg)))
            dev_preds[miss] = np.full(len(dev_avg), float(np.mean(hold_avg)))
            test_preds[miss] = np.full(len(ids_test), float(np.mean(hold_avg)))

    ens = Ensemble("isotonic").fit(holdout_preds, hold_avg)
    ensemble_dev = ens.predict(dev_preds)
    ensemble_test = ens.predict(test_preds)
    write_all_predictions(ids_dev, ids_test, dev_preds, test_preds,
                          ensemble_dev, ensemble_test, PRED_DIR)
    # Persist Component C dev uncertainty for the novelty figure (fig10).
    np.save(PRED_DIR / "_likert_dev_var.npy", np.asarray(dev_var_C))

    dev_off = _score(ids_dev, ensemble_dev, sol_path)
    dev_int_off = _score(ids_dev, ensemble_dev, sol_path, as_int=True)
    gate_dev = uncertainty_gate(dev_preds, dev_var_C, ens.weights)
    results["ensemble"] = {
        "weights": {"A_nli": ens.weights[0], "B_encoder": ens.weights[1],
                    "C_likert": ens.weights[2]},
        "dev_continuous": dev_off,
        "dev_integer": dev_int_off,
        "dev_uncertainty_gated": _score(ids_dev, gate_dev, sol_path),
        "verdict": _verdict(dev_off),
    }
    results["ablations"]["continuous_vs_int"] = {
        "continuous": dev_off, "rounded_int": dev_int_off}
    results["ablations"]["ensemble_drop_one"] = {
        f"drop_{k}": _score(ids_dev, _drop_one(dev_preds, k, ens), sol_path)
        for k in "ABC"
    }
    results["classification"] = M.classification_metrics(ensemble_dev, dev_avg)

    print(f"\n[run_all] ENSEMBLE dev (official): acc@std="
          f"{dev_off['accuracy_at_std']:.4f} rho={dev_off['spearman']:+.4f} "
          f"-> {results['ensemble']['verdict']}")

    results["meta"]["wall_time_sec"] = round(time.time() - t0, 1)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"[run_all] wrote {RESULTS.relative_to(REPO)} "
          f"({results['meta']['wall_time_sec']} s)")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _score(ids, preds, sol_path, as_int=False):
    tmp = PRED_DIR / "_scratch_pred.jsonl"
    D.write_predictions_jsonl(ids, preds, tmp, as_int=as_int)
    s = M.official_score(sol_path, tmp, FINAL / "_scratch_scores.json")
    return {"accuracy_at_std": s["accuracy"], "spearman": s["spearman"]}


def _verdict(score):
    a, r = score["accuracy_at_std"], score["spearman"]
    if a >= 0.80 and r >= 0.70:
        return "Good"
    if a >= 0.70 and r >= 0.60:
        return "OK"
    return "Below OK"


def _drop_one(dev_preds, drop, ens):
    keep = {k: v for k, v in dev_preds.items() if k != drop}
    w = dict(zip("ABC", ens.weights))
    w[drop] = 0.0
    s = sum(w[k] for k in keep) or 1.0
    blend = sum(w[k] / s * keep[k] for k in keep)
    return ens.calibrator.predict(blend)


if __name__ == "__main__":
    main()
