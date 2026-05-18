"""Component A — zero-shot NLI sense-plausibility scorer (no training).

Idea
----
A natural-language-inference model already knows, from large-scale pre-training,
whether a hypothesis is supported by a premise.  We turn "is this word sense
plausible in this story?" into an entailment question:

    premise    = the full 5-sentence story
    hypothesis = 'In this story, the word "<homonym>" means <judged_meaning>.'

We read the model's entailment / contradiction probabilities and define a
scalar plausibility signal

    s = P(entailment) - P(contradiction)   ∈ [-1, 1]

(subtracting contradiction makes the signal monotone and lets the neutral mass
absorb genuine ambiguity).  Two paraphrased hypothesis templates are averaged
for robustness.  ``s`` is then mapped onto the 1–5 Likert scale by a calibrator
fit on **train only** — isotonic regression by default (the NLI score saturates
near ±1, so a monotone non-parametric fit beats a line), with a linear
calibrator kept for the report's calibration ablation.

Because NLI knowledge is homonym-agnostic, this component generalises across the
near homonym-disjoint dev/test split where the milestone's lexical features did
not.  No gradient updates are performed here.
"""
from __future__ import annotations

import numpy as np

# Open NLI checkpoints (downloaded from the HF Hub).
#   PRIMARY  — strong, used on GPU (the canonical / graded configuration).
#   FALLBACK — small & fast 3-class MNLI head, used on the CPU path so the
#              training-free component stays tractable without a GPU.
# GPU path: the strongest open zero-shot NLI checkpoint — a DeBERTa-v3-large
# fine-tuned on MNLI+FEVER+ANLI+LING+WANLI. Far more sense-sensitive than
# bart-large-mnli; label columns are still resolved by text below so the
# checkpoint's id2label ordering does not matter.
PRIMARY_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# CPU path: a strong but CPU-inference-feasible MNLI head (~184M). Far better
# than a distil-MNLI model, and NLI here is inference-only (low RAM, no grads).
FALLBACK_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

HYPOTHESIS_TEMPLATES = (
    'In this story, the word "{homonym}" means {gloss}.',
    'Here, "{homonym}" is used to mean {gloss}.',
)


def _entailment_label_ids(id2label: dict[int, str]) -> tuple[int, int]:
    """Locate the entailment / contradiction columns regardless of label order.

    Different NLI checkpoints order their 3 logits differently
    (MNLI: contradiction/neutral/entailment; BART-MNLI differs), so resolve by
    the textual label rather than a hard-coded index.
    """
    ent = con = None
    for idx, lab in id2label.items():
        low = str(lab).lower()
        if low.startswith("entail"):
            ent = int(idx)
        elif low.startswith("contradict"):
            con = int(idx)
    if ent is None or con is None:  # pragma: no cover - defensive
        raise ValueError(f"cannot find entailment/contradiction in {id2label}")
    return ent, con


def nli_raw_scores(
    stories: list[str],
    homonyms: list[str],
    glosses: list[str],
    *,
    model_name: str = PRIMARY_MODEL,
    device: str | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    n_templates: int = 2,
    progress: bool = True,
) -> np.ndarray:
    """Return the raw plausibility signal ``s`` (one float per example).

    ``n_templates`` averages this many paraphrased hypotheses (2 = canonical;
    the CPU path uses 1 to halve inference cost).  Lazily imports
    torch/transformers so the rest of the package (data, metrics, builders)
    works in an environment without them.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    ent_id, con_id = _entailment_label_ids(model.config.id2label)

    templates = HYPOTHESIS_TEMPLATES[: max(1, int(n_templates))]
    n = len(stories)
    per_template = np.zeros((len(templates), n), dtype=np.float64)

    with torch.no_grad():
        for t_idx, template in enumerate(templates):
            hyps = [
                template.format(homonym=h, gloss=g)
                for h, g in zip(homonyms, glosses)
            ]
            for start in range(0, n, batch_size):
                sl = slice(start, start + batch_size)
                enc = tok(
                    stories[sl],
                    hyps[sl],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                s = probs[:, ent_id] - probs[:, con_id]
                per_template[t_idx, start : start + s.shape[0]] = (
                    s.float().cpu().numpy()
                )
                if progress and (start // batch_size) % 20 == 0:
                    print(
                        f"  NLI [{model_name.split('/')[-1]}] "
                        f"template {t_idx + 1}/{len(templates)} "
                        f"{min(start + batch_size, n)}/{n}",
                        flush=True,
                    )

    # free GPU memory for the next component
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return per_template.mean(axis=0)


class Calibrator:
    """Maps the raw NLI signal onto [1, 5].

    ``method='isotonic'`` (default) — monotone, non-parametric; robust to the
    sigmoidal saturation of NLI probabilities.
    ``method='linear'`` — degree-1 polynomial; kept for the calibration ablation.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._model = None
        self._coef = None

    def fit(self, raw: np.ndarray, avg: np.ndarray) -> "Calibrator":
        raw = np.asarray(raw, float)
        avg = np.asarray(avg, float)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(
                y_min=1.0, y_max=5.0, out_of_bounds="clip"
            ).fit(raw, avg)
        elif self.method == "linear":
            self._coef = np.polyfit(raw, avg, 1)
        else:  # pragma: no cover
            raise ValueError(self.method)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self.method == "isotonic":
            out = self._model.predict(raw)
        else:
            out = np.polyval(self._coef, raw)
        return np.clip(out, 1.0, 5.0)
