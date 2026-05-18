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
        # First token ([CLS]/<s>) representation as the pooled summary.
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


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
    optim = torch.optim.AdamW(
        model.module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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
                loss = loss_fn(pred, tgt) / cfg.grad_accum
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
