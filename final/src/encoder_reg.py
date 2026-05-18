"""Component B — fine-tuned gloss-informed encoder regressor.

A transformer encoder (``microsoft/deberta-v3-large`` on GPU,
``roberta-base`` as the CPU/FAST fallback) is fine-tuned to predict the human
*average* rating directly.  The input is the gloss-informed sentence pair from
:func:`data.build_pair_text` and the head is a single linear layer on the pooled
``[CLS]`` representation trained with MSE.  The regression target is
standardised (z-scored on the training fold) and de-standardised at inference,
which keeps the optimisation well conditioned.

Model selection (early stopping) uses the **homonym-disjoint train hold-out**,
never dev, because the official splits are near homonym-disjoint and a
dev-selected model would not transfer to the hidden test set.

This module also exposes the shared :class:`PairDataset`, :func:`make_loaders`,
seeding and device helpers that Component C (``likert_head``) reuses, so the two
trainable components share one tokenisation / batching path.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Reproducibility / device helpers (shared by Components B and C)
# --------------------------------------------------------------------------- #
def set_all_seeds(seed: int = 42) -> None:
    """Seed python / numpy / torch / cuda and request deterministic kernels."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:  # determinism is best-effort: some cuda kernels have no det. variant
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
    torch.backends.cudnn.benchmark = False


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_tools(use_amp: bool):
    """Return ``(autocast_ctx_factory, grad_scaler)`` valid on torch ≥2.x / 5.x.

    Uses the modern ``torch.amp`` API (the ``torch.cuda.amp`` aliases are
    deprecated in recent torch).  When ``use_amp`` is False both are no-ops, so
    the same training code path runs unchanged on CPU.
    """
    import contextlib

    import torch

    if not use_amp:
        return (lambda: contextlib.nullcontext()), \
            torch.amp.GradScaler("cuda", enabled=False)

    # Prefer bf16 on capable GPUs (e.g. A100): full fp32 exponent range means
    # NO gradient scaler is needed, which sidesteps the "unscale FP16 gradients"
    # failure entirely and is faster/more stable. Fall back to fp16 + scaler on
    # older GPUs (params are forced fp32 in the model, so unscale is valid).
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    def autocast():
        return torch.amp.autocast("cuda", dtype=dtype)

    return autocast, torch.amp.GradScaler("cuda", enabled=not bf16)


# --------------------------------------------------------------------------- #
# Rank-alignment auxiliary loss (shared by Components B and C)
# --------------------------------------------------------------------------- #
# The official metric is Spearman ρ (a *ranking* metric), but the supervised
# objectives (B: MSE, C: KL+MSE) only optimise pointwise fit. We add a small
# batch-level Pearson term ``1 - corr(pred, target)`` — a smooth, differentiable
# surrogate for Spearman (one cannot backprop through hard ranks) — so training
# is aligned with evaluation. This is our own methodological contribution; the
# weight is deliberately small because a per-batch (bs≈16) correlation is a
# noisy estimate of the dataset-level ρ and a large weight destabilises the
# well-conditioned MSE/KL term.
LAMBDA_RANK = 0.1


def _batch_pearson(pred, target):
    """Differentiable batch Pearson correlation, computed in fp32.

    Returns a scalar in [-1, 1], or 0.0 when the batch is degenerate
    (``< 2`` elements, or a constant pred/target ⇒ zero variance). The fp32
    cast and the two guards make this safe under bf16/fp16 autocast and for the
    size-1 final batch that ``make_loaders`` (no ``drop_last``) can yield —
    without them a single NaN here would poison the rest of the run.
    """
    p = pred.float().reshape(-1)
    t = target.float().reshape(-1)
    if p.numel() < 2:
        return p.new_tensor(0.0)
    p = p - p.mean()
    t = t - t.mean()
    denom = p.norm() * t.norm()
    if denom < 1e-8:
        return p.new_tensor(0.0)
    return (p * t).sum() / denom


def build_llrd_groups(module, base_lr, weight_decay, decay: float = 0.95):
    """AdamW parameter groups with layer-wise learning-rate decay.

    Names are introspected by string prefix (``\\.layer\\.<i>\\.``) rather than
    by class/attribute, so it works through the ``ModuleDict({"b": backbone,
    "h": head})`` wrapper and across encoder families. The top encoder layer and
    the head keep ``base_lr``; each lower layer is multiplied by ``decay`` and
    embeddings / relative-position params sit lowest. Bias and LayerNorm get no
    weight decay. If no transformer layers can be identified the function
    degrades to a single flat-LR group — i.e. exactly the previous behaviour —
    so it can never crash an unattended run. Returns ``(groups, used_llrd)``.
    """
    import re

    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    layer_re = re.compile(r"\.layer\.(\d+)\.")
    layer_ids = {
        int(m.group(1)) for n, _ in named for m in [layer_re.search(n)] if m
    }
    if not layer_ids:  # unknown naming → safe flat-LR fallback
        return (
            [{"params": [p for _, p in named], "lr": base_lr,
              "weight_decay": weight_decay}],
            False,
        )
    top = max(layer_ids)
    groups = []
    for n, p in named:
        if n.startswith("h."):  # task head
            lr = base_lr
        else:
            m = layer_re.search(n)
            if m:
                lr = base_lr * (decay ** (top - int(m.group(1))))
            else:  # embeddings / rel_embeddings / final norm → lowest lr
                lr = base_lr * (decay ** (top + 1))
        no_decay = any(
            t in n for t in ("bias", "LayerNorm.weight", "LayerNorm.bias")
        )
        groups.append({"params": [p], "lr": lr,
                       "weight_decay": 0.0 if no_decay else weight_decay})
    return groups, True


@dataclass
class TrainConfig:
    """All knobs for a single training run.

    The GPU defaults target ``deberta-v3-large``; ``run_all`` rewrites
    ``model_name`` / ``epochs`` / ``seeds`` for the CPU / FAST path.
    """

    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 256
    lr: float = 1e-5
    batch_size: int = 16
    grad_accum: int = 1
    epochs: int = 4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seeds: tuple[int, ...] = (13, 42, 123)
    fp16: bool = True
    device: str = field(default_factory=pick_device)


# --------------------------------------------------------------------------- #
# Shared dataset / loaders
# --------------------------------------------------------------------------- #
class PairDataset:
    """Tokenises the gloss-informed sentence pair once and serves tensors.

    ``targets`` is generic: a scalar (Component B) or a length-5 distribution
    (Component C).  ``None`` for the unlabeled test split.
    """

    def __init__(self, seg_a, seg_b, tokenizer, max_length: int, targets=None):
        self.enc = tokenizer(
            list(seg_a),
            list(seg_b),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.targets = None if targets is None else np.asarray(targets, np.float32)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.targets is not None:
            item["target"] = self.targets[i]
        return item


def make_loaders(dataset, tokenizer, batch_size: int, shuffle: bool, seed: int):
    """DataLoader with dynamic padding; deterministic (num_workers=0)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    base = DataCollatorWithPadding(tokenizer)

    def collate(features):
        targets = None
        if "target" in features[0]:
            targets = torch.tensor(
                np.stack([f.pop("target") for f in features]), dtype=torch.float32
            )
        batch = base(features)
        if targets is not None:
            batch["target"] = targets
        return batch

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=g,
    )


# --------------------------------------------------------------------------- #
# Component B model + training
# --------------------------------------------------------------------------- #
class _Regressor:
    """Encoder backbone + 1-unit linear head (built lazily to defer torch)."""

    def __init__(self, model_name: str):
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = cfg.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.module = nn.ModuleDict({"b": self.backbone, "h": self.head})
        # Keep master weights in fp32 (some checkpoints load as fp16 under newer
        # transformers); mixed precision is applied via autocast at compute time.
        self.module.float()

    def forward(self, **batch):
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        # Attention-masked mean pooling. DeBERTa-v3 (and RoBERTa) do not
        # pre-train the first token as a sentence summary, so mean-pooling over
        # the real tokens is a more robust regression representation (Reimers &
        # Gurevych, EMNLP 2019) and reduces seed variance vs. raw [CLS].
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        return self.head(pooled).squeeze(-1)


def _train_one_seed(cfg: TrainConfig, data, seed: int, log_prefix: str):
    """Train a single regressor; early-stop on hold-out Spearman.

    ``data`` carries pre-built seg_a/seg_b text and label arrays for the
    fit / hold-out / dev / test folds (see :func:`run_all`).  Returns dev & test
    continuous predictions plus the per-epoch hold-out curve for the figures.
    """
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from .metrics import spearman

    set_all_seeds(seed)
    device = cfg.device
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    y_fit = np.asarray(data["fit_avg"], np.float32)
    mu, sigma = float(y_fit.mean()), float(y_fit.std() + 1e-6)

    ds_fit = PairDataset(
        data["fit_a"], data["fit_b"], tok, cfg.max_length, (y_fit - mu) / sigma
    )
    ds_hold = PairDataset(data["hold_a"], data["hold_b"], tok, cfg.max_length)
    ds_dev = PairDataset(data["dev_a"], data["dev_b"], tok, cfg.max_length)
    ds_test = PairDataset(data["test_a"], data["test_b"], tok, cfg.max_length)

    dl_fit = make_loaders(ds_fit, tok, cfg.batch_size, True, seed)
    dl_hold = make_loaders(ds_hold, tok, cfg.batch_size, False, seed)
    dl_dev = make_loaders(ds_dev, tok, cfg.batch_size, False, seed)
    dl_test = make_loaders(ds_test, tok, cfg.batch_size, False, seed)

    model = _Regressor(cfg.model_name)
    model.module.to(device)
    groups, used_llrd = build_llrd_groups(
        model.module, cfg.lr, cfg.weight_decay
    )
    optim = torch.optim.AdamW(groups)
    print(
        f"  {log_prefix} seed={seed} "
        f"LLRD={'on' if used_llrd else 'flat-LR fallback'} "
        f"({len(groups)} param groups)",
        flush=True,
    )
    steps = (len(dl_fit) // cfg.grad_accum) * cfg.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(cfg.warmup_ratio * steps), steps
    )
    use_amp = cfg.fp16 and device == "cuda"
    autocast, scaler = amp_tools(use_amp)
    loss_fn = torch.nn.MSELoss()

    def predict(dl):
        model.module.eval()
        preds = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items() if k != "target"}
                with autocast():
                    out = model.forward(**batch)
                preds.append(out.float().cpu().numpy())
        return np.concatenate(preds) * sigma + mu  # de-standardise

    best_rho, best = -2.0, None
    curve = []
    for epoch in range(cfg.epochs):
        model.module.train()
        optim.zero_grad()
        running = 0.0
        for step, batch in enumerate(dl_fit):
            tgt = batch.pop("target").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                pred = model.forward(**batch)
                # Assemble the composite loss BEFORE the grad_accum division so
                # the MSE and rank terms stay correctly proportioned.
                base = loss_fn(pred, tgt)
                rank = 1.0 - _batch_pearson(pred, tgt)
                loss = (base + LAMBDA_RANK * rank) / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), cfg.max_grad_norm
                )
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()
        hold_pred = predict(dl_hold)
        rho = spearman(hold_pred, data["hold_avg"])
        curve.append(rho)
        print(
            f"  {log_prefix} seed={seed} epoch {epoch + 1}/{cfg.epochs} "
            f"train_mse={running / max(1, len(dl_fit)):.4f} "
            f"holdout_rho={rho:+.4f}",
            flush=True,
        )
        # Capture dev/test AND hold-out predictions at the best epoch so the
        # ensemble weight search reuses this hold-out vector — no extra retrain.
        if rho > best_rho:
            best_rho = rho
            best = (predict(dl_dev), predict(dl_test), hold_pred)

    if best is None:  # epochs==0 guard
        best = (predict(dl_dev), predict(dl_test), predict(dl_hold))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "dev": np.clip(best[0], 1, 5),
        "test": np.clip(best[1], 1, 5),
        "holdout": np.clip(best[2], 1, 5),
        "holdout_curve": curve,
        "best_holdout_rho": best_rho,
    }


def train_regressor(cfg: TrainConfig, data, log_prefix: str = "EncReg"):
    """Train across ``cfg.seeds`` and average the continuous predictions."""
    devs, tests, holds, curves, rhos = [], [], [], [], []
    for seed in cfg.seeds:
        r = _train_one_seed(cfg, data, seed, log_prefix)
        devs.append(r["dev"])
        tests.append(r["test"])
        holds.append(r["holdout"])
        curves.append(r["holdout_curve"])
        rhos.append(r["best_holdout_rho"])
    return {
        "dev": np.mean(devs, axis=0),
        "test": np.mean(tests, axis=0),
        "holdout": np.mean(holds, axis=0),
        "holdout_curves": curves,
        "best_holdout_rho": float(np.mean(rhos)),
        "seeds": list(cfg.seeds),
    }
