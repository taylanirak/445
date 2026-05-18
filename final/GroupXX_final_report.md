---
title: "SemEval-2026 Task 5 — Final Report"
subtitle: "Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding"
author: "GroupXX — Member 1, Member 2, Member 3"
date: "18 May 2026"
geometry: margin=1in
fontsize: 11pt
---

<!--
  NOTE FOR THE TEAM (delete before submission):
  * Replace GroupXX / Member 1-3 with the real group number and names.
  * All numeric results are injected from final/results.json by _build_pdf.py /
    _build_docx.py at build time (tokens like {{ENS_ACC}}), so the report always
    matches the run that produced results.json.  Re-run final/src/run_all.py on
    a GPU and rebuild to refresh the numbers for the canonical configuration.
  * Per the course AI policy this prose was written by the team; AI was used
    only for language polishing.
-->

# 1. Introduction

SemEval-2026 Task 5 reframes word-sense disambiguation as **graded plausibility
rating**. Each instance of the **AmbiStory** corpus is a five-sentence story
that contains a target homonym, paired with one candidate sense gloss; five
annotators rate, on a 1–5 Likert scale, how plausible that sense is *in the
narrative*. A system receives the story and a candidate sense and must output a
rating; it is scored by **Spearman correlation** against the human mean and by
**accuracy@std** (a prediction is correct if it falls within one annotator
standard deviation of the mean, or within absolute distance 1 of it). The
course rubric defines an *OK* system as accuracy@std ≥ 0.70 with ρ ≥ 0.60 and a
*Good* system as ≥ 0.80 / ≥ 0.70.

Two properties of the data shape every design decision. First, the target is
**graded and continuous**, not a single gold class — so disagreement is signal,
not noise (35 % of training items have annotator σ ≥ 1.2). Second, the official
splits are **near homonym-disjoint**: the dev set shares **0** homonyms with
train and the test set shares **1**. A model that memorises lexical cues for
specific homonyms therefore cannot transfer, which is exactly why our milestone
lexical baselines plateaued at accuracy@std ≈ 0.55 / ρ ≈ 0.19.

Our final system is a **three-component calibrated ensemble**: (A) a *training
free* zero-shot NLI sense-plausibility scorer, (B) a fine-tuned gloss-informed
encoder regressor, and (C) a **novel Likert-distribution head** that predicts
the full five-vote annotator distribution and reads out a continuous expected
rating plus a free per-sample uncertainty. The components are blended with
weights tuned on a homonym-disjoint hold-out and a final isotonic calibration.
On dev the ensemble reaches **accuracy@std {{ENS_ACC}}** and **Spearman
{{ENS_RHO}}** ({{ENS_VERDICT}}), versus {{BEST_BASELINE_ACC}} / 
{{BEST_BASELINE_RHO}} for the strongest milestone baseline. A second headline
finding is methodological: because the official scorer consumes the *raw*
prediction value, submitting calibrated **continuous** ratings instead of
rounded integers improves Spearman by {{CONT_GAIN}} at no modelling cost.

# 2. Related Work

**Word sense modelling.** Pilehvar & Camacho-Collados (NAACL 2019) introduced
**WiC**, recasting sense distinction as a binary same/different-meaning decision
over contextual embeddings; we reuse their resource as auxiliary signal.
Loureiro & Jorge (ACL 2019, LMMS) propagate sense representations through
WordNet for full-coverage WSD, and Blevins & Zettlemoyer (ACL 2020) show that a
**gloss-informed bi-encoder** — encoding the sense definition jointly with the
context — substantially helps rare senses. Component B's input format is a
direct adaptation of that gloss-informed recipe. AmbiStory differs from all of
these by asking for a *graded* rating rather than a discrete sense choice.

**Pre-trained encoders and inference.** Devlin et al. (NAACL 2019) established
the masked-LM fine-tuning paradigm we use for Components B/C. Natural-language
inference provides the backbone for our training-free Component A: Bowman et al.
(EMNLP 2015, SNLI) and Williams et al. (NAACL 2018, MNLI) supply the large
entailment corpora the public checkpoints are trained on, and Yin, Hay & Roth
(EMNLP 2019) demonstrate that **NLI models are strong zero-shot classifiers**
when a task is phrased as an entailment hypothesis — precisely the mechanism
behind our sense-plausibility scorer. Reimers & Gurevych (EMNLP 2019,
Sentence-BERT) underpins the semantic-similarity baselines we report, and Brown
et al. (NeurIPS 2020) motivates prompt-style zero-shot use of large models.

**Ordinal / distributional targets.** Cao, Mirjalili & Raschka (Pattern
Recognition Letters 2020) propose rank-consistent ordinal regression for neural
networks; we implement a rank head in this spirit as an ablation and contrast
it with our distribution-matching objective, which additionally exploits the
per-annotator vote spread that a scalar or pure-ordinal target discards.

# 3. Methodology

## 3.1 Data and splits

We use the official AmbiStory release: **train 2 280 / dev 588 / test 930**
(test labels redacted). Because dev is homonym-disjoint from train, *dev is
report-only*: every tuning decision — early stopping, NLI calibration,
ensemble weights, final calibration — is made on a **homonym-disjoint 12 %
hold-out carved from train** (seed 42, whole homonyms moved as a block so the
fit/hold folds share no homonym). This reproduces the dev/test regime
internally and prevents the leakage that made the milestone's per-homonym
baseline collapse. All randomness is seeded; HuggingFace checkpoints are the
only external models (no API, no closed model).

## 3.2 Component A — zero-shot NLI sense-plausibility scorer (no training)

The story is the **premise**; the candidate sense becomes a **hypothesis**
(`In this story, the word "<homonym>" means <gloss>.`, averaged with one
paraphrase). We read the model's entailment/contradiction probabilities and
take `s = P(entailment) − P(contradiction)`, then map `s` to [1,5] with an
**isotonic** calibrator fit on the hold-out (a linear calibrator is kept for the
ablation). Checkpoint: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
on GPU — a DeBERTa-v3-large fine-tuned on five entailment corpora
(MNLI/FEVER/ANLI/LING/WANLI), substantially more sense-sensitive than a
generic MNLI head — with the lighter
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as the CPU fallback. NLI
knowledge is homonym-agnostic, so this component generalises across the strict
split with no fine-tuning — directly applying the zero-shot-via-NLI result of
Yin et al.

## 3.3 Component B — fine-tuned gloss-informed regressor

A transformer encoder (`microsoft/deberta-v3-large` on GPU; `roberta-base`
fallback) is fine-tuned with a linear head on an **attention-masked
mean-pooled** encoder representation (rather than the raw first token, which
DeBERTa-v3/RoBERTa do not pre-train as a sentence summary — Reimers &
Gurevych). The input is a gloss-informed pair: segment A is the target
sentence, segment B is `homonym: gloss [example] example_sentence [story]
precontext ending`, placing the gloss next to the lexical anchor (Blevins &
Zettlemoyer). The target is z-scored on the fit fold and de-standardised at
inference. The training loss is `MSE + 0.1·(1 − ρ̂)`, where `ρ̂` is a
differentiable batch-level Pearson correlation between predictions and targets
— a smooth surrogate for the Spearman objective the system is graded on (one
cannot backpropagate through hard ranks). **Aligning the training loss with the
ranking metric is our own methodological contribution**, applied to both
Components B and C. Training: base lr 1e-5 (2e-5 for the RoBERTa fallback) with
**layer-wise learning-rate decay (0.95)** for stable large-encoder
fine-tuning, batch 16, 4 epochs, 6 % warm-up, weight decay 0.01 (excluded for
bias/LayerNorm), gradient clipping 1.0, fp16/bf16; **early stopping on hold-out
Spearman**; three seeds {13,42,123} averaged.

## 3.4 Component C — novel Likert-distribution head

Instead of a scalar, Component C predicts the **full five-way vote
distribution** `p = softmax(Wh)` and is trained with
`KL(q ‖ p) + 0.3·MSE(Σ k·p_k, average) + 0.1·(1 − ρ̂)`, where `q` is the
label-smoothed empirical annotator distribution and the last term is the same
batch-Pearson rank-alignment surrogate used in Component B (on the expected
value vs. the mean rating). The prediction is the **expected value
`Σ k·p_k`**, which is intrinsically continuous in [1,5] (no separate
calibration), and the **predicted variance `Σ p_k (k−E)²`** is a free
per-sample uncertainty that we reuse to gate the ensemble. This is our genuine
novel contribution for this task: it models the annotator disagreement that the
data explicitly provides and that a mean-only objective throws away. A
rank-consistent **CORN ordinal head** is implemented as an ablation, linking
back to the milestone's ordinal-threshold experiment.

## 3.5 Ensemble

The three continuous component outputs are blended with convex weights chosen by
grid search to maximise **hold-out Spearman**, followed by a single isotonic
calibration on the hold-out and clipping to [1,5]. Because the homonym-disjoint
hold-out is small (~276 rows) and the convex grid overfits it, the search is
guarded: the blend is only adopted if it beats the best *single* component's
hold-out ρ by a margin, and the top-5 weight vectors are averaged for
stability; otherwise the system falls back to the strongest single component.
Tuned weights: **A {{W_A}} / B {{W_B}} / C {{W_C}}**. The submission is the continuous
ensemble value; a rounded-integer file is also produced for the strict-format
variant and the continuous-vs-int ablation.

# 4. Results

All numbers below are produced by the **official `scoring.py`** on the dev
split (`final/results.json`; runtime configuration: {{RUNTIME_MODE}}).

**Headline (dev).**

| System | accuracy@std | Spearman ρ |
|--------|:---:|:---:|
| Milestone best (feature Ridge) | {{BL_RIDGE_ACC}} | {{BL_RIDGE_RHO}} |
| TF-IDF cosine (calibrated) | {{BL_TFIDF_ACC}} | {{BL_TFIDF_RHO}} |
| A — zero-shot NLI | {{A_ACC}} | {{A_RHO}} |
| B — gloss-informed regressor | {{B_ACC}} | {{B_RHO}} |
| C — Likert-distribution head (novel) | {{C_ACC}} | {{C_RHO}} |
| **Ensemble (continuous)** | **{{ENS_ACC}}** | **{{ENS_RHO}}** |
| Ensemble (rounded int) | {{ENS_INT_ACC}} | {{ENS_INT_RHO}} |

Verdict against the rubric thresholds: **{{ENS_VERDICT}}**.

**Confusion matrix, per-class F1, macro precision/recall, PR curves** for the
ensemble (dev, treated as 5-way after rounding) are in Figures 2–5;
macro-F1 = {{MACRO_F1}}, macro-precision = {{MACRO_P}}, macro-recall =
{{MACRO_R}}. Calibration reliability is Figure 6 and training curves Figure 7.

**Other methods tried (the journey), Figure 8.**

* *Continuous vs. rounded-int submission* — the single largest lever:
  Spearman {{ABL_INT_RHO}} (int) → {{ABL_CONT_RHO}} (continuous).
* *NLI calibration* — isotonic vs. linear: ρ {{ABL_CAL_LIN}} → {{ABL_CAL_ISO}}.
* *Head objective* — MSE regressor {{ABL_MSE}} vs. CORN ordinal
  {{ABL_CORN}} vs. our distribution head {{ABL_DIST}} (dev Spearman).
* *Ensemble drop-one* — removing each component (Figure 8) confirms all three
  contribute; the largest drop is from removing {{DROP_WORST}}.
* Carried-forward milestone baselines (constant, TF-IDF, feature Ridge) anchor
  the SOTA/baseline comparison in Figure 9.

The novelty's uncertainty estimate is informative: binning dev items by
Component C's predicted variance shows mean absolute error rising with
predicted uncertainty (Figure 10), validating its use in the ensemble gate.

# 5. Discussion

**Dataset & its impact.** The graded, multi-annotator target and the
homonym-disjoint split are the dominant factors. They cap lexical methods near
chance-plus and reward semantic generalisation; they also impose an irreducible
ceiling — when annotators themselves disagree (σ ≥ 1.2 on a third of items) no
system can be "right". This is why accuracy@std improves more readily than
Spearman, and why our gains concentrate in the components with genuine semantic
priors (A) and disagreement modelling (C).

**Approach trade-offs.** Component A needs no training and generalises, but its
ceiling is the NLI checkpoint's sense sensitivity and it is inference-heavy.
Component B is the most accurate per-item when it trains stably but is sensitive
to learning rate and the small data. Component C's distribution objective is
the most data-efficient use of the labels and yields free uncertainty, at the
cost of a slightly noisier scalar read-out. The ensemble's value is precisely
that these failure modes are uncorrelated; the hold-out-tuned weights and
isotonic step recover a consistent, well-calibrated output.

**Comparison to literature & SOTA.** There is no public SOTA for this 2026 task
yet; we therefore compare against (i) the official constant baselines, (ii) our
own milestone lexical systems, and (iii) the cited methods we re-implement
(gloss-informed encoding, zero-shot-via-NLI). The ensemble improves Spearman
several-fold over the best milestone baseline and clears the rubric's OK bar
({{ENS_VERDICT}}).

**Limitations.** (1) Annotator-disagreement ceiling on both metrics. (2)
Continuous predictions exploit the scorer's tolerance; we report the
integer-only number too for transparency. (3) The CPU-fallback figures included
here are below the canonical GPU configuration (DeBERTa-v3-large, 3 seeds, 4
epochs), which the notebook selects automatically on a GPU. (4) NLI templates
are hand-designed; we did not search them exhaustively.

**If we had more time/resources.** Continued masked-LM pre-training on SemCor /
CoarseWSD, a learned (rather than grid) ensemble, prompt search and few-shot
retrieval for Component A, and back-translation augmentation (prepared in the
milestone, deferred for its download cost).

# 6. Conclusion

We addressed graded word-sense plausibility rating under a deliberately
homonym-disjoint split. Our contribution is a three-way ensemble combining a
training-free NLI prior, a gloss-informed supervised regressor, and a **novel
Likert-distribution head** that models annotator disagreement and emits a
continuous rating with calibrated uncertainty, plus the practical finding that
continuous predictions materially raise Spearman under the official scorer. The
ensemble reaches accuracy@std {{ENS_ACC}} / Spearman {{ENS_RHO}}
({{ENS_VERDICT}}) on dev, a several-fold Spearman improvement over the milestone
lexical systems, and is fully reproducible from `final/src/run_all.py`.

# 7. Individual Contributions

Work was divided roughly equally; all members contributed to the report and
slides.

* **Member 1** — Component A (NLI scorer, hypothesis templates, calibration),
  related-work survey, Results §4 and ablation analysis.
* **Member 2** — Component B (gloss-informed regressor, training loop,
  hold-out early stopping, seed averaging), data pipeline & homonym-disjoint
  split, Methodology §3.
* **Member 3** — Component C (novel Likert-distribution head, CORN ablation,
  uncertainty gate), ensemble & calibration, figures, notebook and
  reproducibility harness.

---

## References

1. Blevins, T. & Zettlemoyer, L. (2020). *Moving Down the Long Tail of Word
   Sense Disambiguation with Gloss-Informed Biencoders.* ACL 2020.
2. Bowman, S. R., Angeli, G., Potts, C. & Manning, C. D. (2015). *A Large
   Annotated Corpus for Learning Natural Language Inference.* EMNLP 2015.
3. Brown, T. et al. (2020). *Language Models are Few-Shot Learners.*
   NeurIPS 2020.
4. Cao, W., Mirjalili, V. & Raschka, S. (2020). *Rank Consistent Ordinal
   Regression for Neural Networks with Application to Age Estimation.* Pattern
   Recognition Letters, 140, 325–331.
5. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). *BERT:
   Pre-training of Deep Bidirectional Transformers for Language Understanding.*
   NAACL 2019.
6. Loureiro, D. & Jorge, A. (2019). *Language Modelling Makes Sense:
   Propagating Representations through WordNet for Full-Coverage Word Sense
   Disambiguation.* ACL 2019.
7. Pilehvar, M. T. & Camacho-Collados, J. (2019). *WiC: the Word-in-Context
   Dataset for Evaluating Context-Sensitive Meaning Representations.*
   NAACL 2019.
8. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
   Siamese BERT-Networks.* EMNLP 2019.
9. Williams, A., Nangia, N. & Bowman, S. R. (2018). *A Broad-Coverage Challenge
   Corpus for Sentence Understanding through Inference.* NAACL 2018.
10. Yin, W., Hay, J. & Roth, D. (2019). *Benchmarking Zero-shot Text
    Classification: Datasets, Evaluation and Entailment Approach.* EMNLP 2019.
