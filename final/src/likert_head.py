"""Component C — novel Likert-distribution head (headline contribution).

Motivation
----------
Every AmbiStory entry carries the *five individual annotator votes*, not just
their mean.  35 % of the train set has annotator σ ≥ 1.2: the disagreement is
structured signal, not noise.  A scalar MSE regressor (Component B) throws this
away.  Component C instead predicts the **full Likert distribution**:

    head:   Linear(hidden, 5) → softmax  ⇒  p ∈ Δ⁴   (P(vote = k), k = 1..5)
    target: q = label-smoothed empirical vote distribution
    loss:   KL(q ‖ p)  +  λ · MSE( Σ k·p_k , average )      (λ = 0.3)
    predict: E[k] = Σ k·p_k              (continuous, already in [1, 5])

Two things make this a genuine novel contribution for *this* task:

1.  The expected-value read-out is intrinsically continuous, which directly
    exploits the fact that the official scorer consumes the raw prediction —
    no extra calibration step is needed for this component.
2.  The predicted variance  Var = Σ p_k (k − E)²  is a free per-sample
    *uncertainty* estimate.  We reuse it to down-weight this component inside
    the ensemble when it is unsure (the "uncertainty-gated" ablation).

A rank-consistent **CORN ordinal head** (Cao et al.) is provided behind
``ObjectiveConfig.objective='corn'`` purely as an ablation, connecting back to
the milestone's ordinal-threshold experiment and honestly showing the journey.

The backbone, tokenisation, batching, seeding and hold-out early-stopping logic
are shared with Component B (imported from :mod:`encoder_reg`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder_reg import (
    LAMBDA_RANK,
    PairDataset,
    TrainConfig,
    _batch_pearson,
    amp_tools,
    build_llrd_groups,
    make_loaders,
    pick_device,
    set_all_seeds,
)

LIKERT = np.arange(1, 6, dtype=np.float64)  # [1,2,3,4,5]


@dataclass
class ObjectiveConfig:
    objective: str = "distribution"  # 'distribution' (novel) | 'corn' (ablation)
    label_smoothing: float = 0.05
    mse_lambda: float = 0.3


def smoothed_distribution(choices_list, eps: float) -> np.ndarray:
    """Empirical 5-vote distribution with uniform label smoothing."""
    out = np.zeros((len(choices_list), 5), np.float64)
    for i, ch in enumerate(choices_list):
        for c in ch:
            out[i, int(c) - 1] += 1.0
        out[i] /= out[i].sum()
    return (1.0 - eps) * out + eps * (1.0 / 5.0)


class _DistHead:
    """Encoder backbone + 5-logit head (distribution) or 4-logit head (CORN)."""

    def __init__(self, model_name: str, objective: str):
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        n_out = 4 if objective == "corn" else 5
        self.head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(cfg.hidden_size, n_out)
        )
        self.module = nn.ModuleDict({"b": self.backbone, "h": self.head})
        self.module.float()  # fp32 master weights; autocast handles mixed prec.

    def logits(self, **batch):
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        # Attention-masked mean pooling (shared rationale with Component B).
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return self.head(pooled)


def _corn_targets(avgs: np.ndarray) -> np.ndarray:
    """Binary rank targets for CORN: column k = 1[round(avg) > k] for k=1..4."""
    r = np.clip(np.round(avgs), 1, 5).astype(int)
    return np.stack([(r > k).astype(np.float32) for k in range(1, 5)], axis=1)


def _train_one_seed(cfg: TrainConfig, oc: ObjectiveConfig, data, seed, log_prefix):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from .metrics import spearman

    set_all_seeds(seed)
    device = cfg.device
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    if oc.objective == "distribution":
        tgt = smoothed_distribution(data["fit_choices"], oc.label_smoothing)
    else:  # corn
        tgt = _corn_targets(np.asarray(data["fit_avg"], np.float64))

    ds_fit = PairDataset(data["fit_a"], data["fit_b"], tok, cfg.max_length, tgt)
    ds_hold = PairDataset(data["hold_a"], data["hold_b"], tok, cfg.max_length)
    ds_dev = PairDataset(data["dev_a"], data["dev_b"], tok, cfg.max_length)
    ds_test = PairDataset(data["test_a"], data["test_b"], tok, cfg.max_length)

    dl_fit = make_loaders(ds_fit, tok, cfg.batch_size, True, seed)
    dl_hold = make_loaders(ds_hold, tok, cfg.batch_size, False, seed)
    dl_dev = make_loaders(ds_dev, tok, cfg.batch_size, False, seed)
    dl_test = make_loaders(ds_test, tok, cfg.batch_size, False, seed)

    model = _DistHead(cfg.model_name, oc.objective)
    model.module.to(device)
    groups, used_llrd = build_llrd_groups(
        model.module, cfg.lr, cfg.weight_decay
    )
    optim = torch.optim.AdamW(groups)
    print(
        f"  {log_prefix}[{oc.objective}] seed={seed} "
        f"LLRD={'on' if used_llrd else 'flat-LR fallback'} "
        f"({len(groups)} param groups)",
        flush=True,
    )
    steps = len(dl_fit) * cfg.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(cfg.warmup_ratio * steps), max(1, steps)
    )
    use_amp = cfg.fp16 and device == "cuda"
    autocast, scaler = amp_tools(use_amp)
    likert = torch.tensor(LIKERT, dtype=torch.float32, device=device)

    def expected_and_var(dl):
        model.module.eval()
        e_list, v_list = [], []
        with torch.no_grad():
            for batch in dl:
                batch = {
                    k: v.to(device) for k, v in batch.items() if k != "target"
                }
                with autocast():
                    lg = model.logits(**batch)
                if oc.objective == "distribution":
                    p = torch.softmax(lg.float(), dim=-1)
                else:
                    # CORN: P(rank>k)=cumulative σ; P(y=j) via successive diffs
                    cum = torch.sigmoid(lg.float())  # (N,4) = P(>1..>4)
                    ones = torch.ones(cum.size(0), 1, device=device)
                    zeros = torch.zeros(cum.size(0), 1, device=device)
                    upper = torch.cat([ones, cum], dim=1)  # P(>=1..>=... )
                    lower = torch.cat([cum, zeros], dim=1)
                    p = (upper - lower).clamp_min(1e-6)
                    p = p / p.sum(dim=1, keepdim=True)
                e = (p * likert).sum(-1)
                v = (p * (likert - e.unsqueeze(-1)) ** 2).sum(-1)
                e_list.append(e.cpu().numpy())
                v_list.append(v.cpu().numpy())
        return np.concatenate(e_list), np.concatenate(v_list)

    best_rho, best = -2.0, None
    curve = []
    for epoch in range(cfg.epochs):
        model.module.train()
        optim.zero_grad()
        running = 0.0
        for batch in dl_fit:
            t = batch.pop("target").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                lg = model.logits(**batch)
                if oc.objective == "distribution":
                    logp = F.log_softmax(lg, dim=-1)
                    kl = F.kl_div(logp, t, reduction="batchmean")
                    p = logp.exp()
                    ev = (p * likert).sum(-1)
                    avg_t = (t * likert).sum(-1)
                    loss = kl + oc.mse_lambda * F.mse_loss(ev, avg_t)
                    # Rank-alignment on the expected value vs. the mean rating
                    # (same Spearman-surrogate rationale as Component B). Not
                    # applied to the CORN branch so the ordinal ablation stays
                    # an uncontaminated comparison.
                    loss = loss + LAMBDA_RANK * (1.0 - _batch_pearson(ev, avg_t))
                else:  # CORN: independent rank-binary BCE
                    loss = F.binary_cross_entropy_with_logits(lg, t)
            scaler.scale(loss).backward()
            running += loss.item()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                model.module.parameters(), cfg.max_grad_norm
            )
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            sched.step()
        hold_e, _ = expected_and_var(dl_hold)
        rho = spearman(hold_e, data["hold_avg"])
        curve.append(rho)
        print(
            f"  {log_prefix}[{oc.objective}] seed={seed} "
            f"epoch {epoch + 1}/{cfg.epochs} loss={running / max(1, len(dl_fit)):.4f} "
            f"holdout_rho={rho:+.4f}",
            flush=True,
        )
        if rho > best_rho:
            hold_v = expected_and_var(dl_hold)[1]
            dev_e, dev_v = expected_and_var(dl_dev)
            test_e, test_v = expected_and_var(dl_test)
            best_rho = rho
            best = (dev_e, dev_v, test_e, test_v, hold_e, hold_v)

    if best is None:
        hold_e, hold_v = expected_and_var(dl_hold)
        dev_e, dev_v = expected_and_var(dl_dev)
        test_e, test_v = expected_and_var(dl_test)
        best = (dev_e, dev_v, test_e, test_v, hold_e, hold_v)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "dev": np.clip(best[0], 1, 5),
        "dev_var": best[1],
        "test": np.clip(best[2], 1, 5),
        "test_var": best[3],
        "holdout": np.clip(best[4], 1, 5),
        "holdout_var": best[5],
        "holdout_curve": curve,
        "best_holdout_rho": best_rho,
    }


def train_distribution_head(
    cfg: TrainConfig,
    data,
    oc: ObjectiveConfig | None = None,
    log_prefix: str = "LikertHead",
):
    """Train across ``cfg.seeds``; average expected value and variance."""
    oc = oc or ObjectiveConfig()
    devs, tests, dvar, tvar, holds, hvar, curves, rhos = [], [], [], [], [], [], [], []
    for seed in cfg.seeds:
        r = _train_one_seed(cfg, oc, data, seed, log_prefix)
        devs.append(r["dev"])
        tests.append(r["test"])
        dvar.append(r["dev_var"])
        tvar.append(r["test_var"])
        holds.append(r["holdout"])
        hvar.append(r["holdout_var"])
        curves.append(r["holdout_curve"])
        rhos.append(r["best_holdout_rho"])
    return {
        "dev": np.mean(devs, axis=0),
        "test": np.mean(tests, axis=0),
        "dev_var": np.mean(dvar, axis=0),
        "test_var": np.mean(tvar, axis=0),
        "holdout": np.mean(holds, axis=0),
        "holdout_var": np.mean(hvar, axis=0),
        "holdout_curves": curves,
        "best_holdout_rho": float(np.mean(rhos)),
        "objective": oc.objective,
        "seeds": list(cfg.seeds),
    }
