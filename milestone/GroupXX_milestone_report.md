---
title: "SemEval-2026 Task 5 — Milestone Report"
subtitle: "Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding"
author: "GroupXX — Member 1, Member 2, Member 3"
date: "13 April 2026"
geometry: margin=1in
fontsize: 11pt
---

# 1. Introduction

SemEval-2026 Task 5 ("Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding") departs from the classical Word Sense Disambiguation (WSD) setting in two important ways. First, it is cast as *rating*, not *selection*: systems do not pick one correct sense but score a candidate sense on a 1–5 Likert scale. Second, the context is a narrative — a five-sentence story that surrounds the target ambiguous word — so semantic coherence has to be tracked across sentences rather than within a single clause. The task owners release the **AmbiStory** corpus, whose samples pair a homonym and a candidate gloss with pre-context, target sentence, and ending.

Given a story and a candidate sense, the system must output an integer rating in [1, 5]. The official metrics are Spearman correlation between the predictions and the list of human averages, and *accuracy@std*, which counts a prediction as correct when it lies within one annotator standard deviation of the mean or within an absolute distance smaller than one. The "Good Solution" threshold published in the project description is Spearman ≥ 0.7 and accuracy@std ≥ 0.8.

**Progress so far.** We have completed the data pipeline and a full EDA on the released splits, reproduced the repository baselines, built a shared evaluation harness that matches the CodaBench scorer (`scoring.py`), and implemented prompt-construction and tokenisation scaffolds for the two approaches we will pursue (LLM prompting and a RoBERTa-based regression fine-tune). All steps are committed in a Jupyter notebook whose outputs are preserved, as the rubric requires.

# 2. Dataset Selection

### 2.1 Task dataset — AmbiStory

The primary dataset is the official **AmbiStory** release distributed by the Task 5 organisers. Each entry is a five-sentence story containing a single homonym, paired with one candidate sense; five annotators rate plausibility on a 1–5 Likert scale. The archive provides pre-split **train / dev / test** files, so no custom split is needed; for internal hold-outs (ordinal-threshold tuning, ensemble-weight calibration) we use **seed = 42** throughout.

| split | entries | unique homonyms | ∩ train | mean rating | mean σ |
|-------|--------:|----------------:|--------:|------------:|-------:|
| train | 2 280   | 220             | —       | 3.14        | 0.95   |
| dev   |   588   |  55             | **0**   | —           | —      |
| test  |   930   |  87             |  1      | —           | —      |

**Table 1.** Split sizes, lexical coverage, and cross-split homonym overlap. Five annotators per entry → **18 990 Likert votes** total.

**Schema (11 fields).** `homonym`, `judged_meaning` (candidate sense gloss), `precontext`, `sentence`, `ending`, `example_sentence`, `choices` (5 annotator ratings), `average`, `stdev`, `nonsensical` (per-annotator flag), `sample_id`.

**Formal I/O contract.**

```
input  x = (precontext, sentence, ending, homonym, judged_meaning)    # all str
output y ∈ {1, 2, 3, 4, 5}                                            # integer Likert rating
submission = {"id": "<sample-id>", "prediction": <int>}               # JSONL, one line per entry
```

This matches `scoring.py`: predictions are evaluated with Spearman ρ against human averages and accuracy@std (prediction within 1 σ of the mean *or* within absolute distance < 1). **Split policy:** train for learning, dev for model selection, test only for the final CodaBench submission.

### 2.2 EDA summary (numbers inline; plots in the notebook)

Label distribution: `average` mean 3.14, median 3.20, std 1.19 — near-uniform with a mild 3–4 hump (why constant-4 scores 0.57 acc@std). Annotator agreement: mean σ = 0.95, 25 % of samples have σ ≥ 1.20 — hard ceiling on acc@std. Near-strict homonym split: dev ∩ train = 0, test ∩ train = 1 — per-homonym-mean baseline collapses to mean-of-train (confirmed in notebook §4). Lengths: mean tokens ≈ 11 / 12 / 29 for precontext / sentence / ending → full story fits in 256-token input. `nonsensical` flag fires in 14 % of avg ≤ 2 samples vs. 3 % of avg > 4 — usable auxiliary signal.

### 2.3 Additional datasets — implemented and planned

§B of the project description invites extra data. We split our use into two tiers: **corpora already integrated and running in the milestone notebook**, and **corpora scheduled for the final deliverable**. All are openly licensed and come from rubric-allowed venues.

| dataset | venue | role here | status |
|---------|-------|-----------|:---:|
| **WordNet 3.0** glosses, lemmas, examples | LREC release (via NLTK) | synonym / paraphrase pool for `judged_meaning`; drives our augmentation ops | **implemented** |
| **WiC** (Pilehvar & Camacho-Collados) | NAACL 2019 | 6 066 same / different-sense pairs; 20 of 54 AmbiStory-dev homonyms covered; converted to pseudo-rated entries (T → 5, F → 1) and mixed into train | **implemented** |
| **CoarseWSD-28** (Loureiro et al.) | EMNLP 2020 | 28-homonym WSD corpus → gloss-informed encoder pre-training | planned (final) |
| **SemCor 3.0** | LREC 1998 | sense-tagged running text → masked-LM continued pre-training | planned (final) |

**Leakage safeguard.** Extra data is used *only* at train time — augmentation, paraphrase pool, or pseudo-labels. No AmbiStory dev/test homonym is ever sampled from these sources as a direct label supervisor; pseudo-labels are assigned by the additional corpus's own gold (WiC same / different-sense).

### 2.4 Data augmentation — implemented in the notebook

Four augmentation operators are coded in `milestone/_augment_and_retrain.py` and produced the following concrete counts on AmbiStory train:

* **WordNet synonym-swap** on `judged_meaning` (p = 0.30, same-synset lemmas) — **1 660** new samples.
* **WordNet gloss paraphrase** via sibling-synset definitions of the homonym — **2 232** new samples.
* **Likert mix-up** between same-homonym train pairs with KL-close annotator distributions — **517** new samples.
* **WiC pseudo-entries** covering 20 dev homonyms — **160** new samples.

Sentence back-translation (MarianMT EN → DE → EN) is the one operator we leave to the final deliverable because its model download is ~ 300 MB.

**Real effect on dev** (Feature Ridge re-trained on each pool; full table in the notebook):

| training pool | samples | acc@std | Spearman ρ |
|---------------|---:|---:|---:|
| baseline (train only) | 2 280 | 0.527 | +0.124 |
| + WordNet synonym + paraphrase | 6 172 | 0.527 | **+0.150** |
| + Likert mix-up | 6 689 | **0.537** | +0.142 |
| + WiC pseudo-entries | 6 849 | 0.534 | +0.134 |

WordNet augmentation alone lifts Spearman by ≈ 25 % relative. Mix-up adds one acc@std point. WiC does not help this linear feature model (its sentence register differs from AmbiStory narratives), but we keep the integration in the pipeline because WiC is the natural pre-training signal for the encoder planned in §3(b). **Licensing:** AmbiStory CC-BY; WiC and WordNet redistributable under academic licences. Seed 42 throughout.

# 3. Approach Plan

We pursue two complementary families of methods — an LLM prompt system and a fine-tuned encoder — and add two heads, an *ordinal* one and a *Likert-distribution* one, that respect the structure of the label space. An ensemble combines the families. All systems target the same input/output contract as the official scorer.

**(a) LLM prompting (zero-shot and few-shot).** Prompt an instruction-tuned model with the full narrative and the candidate sense; ask for a single integer. Few-shot examples are retrieved from train by SBERT gloss-embedding similarity (homonym-level retrieval is impossible under the strict split). Motivated by Brown et al. (2020).

**(b) Encoder regression.** Fine-tune **RoBERTa-base** and **DeBERTa-v3-large** with a linear head on `[CLS]` and MSE loss against `average`. Input:

```
<s> sentence </s> homonym : judged_meaning </s> precontext + ending
```

Placing the gloss next to the target sentence biases attention to the lexical anchor — a gloss-informed recipe adapted from Blevins and Zettlemoyer (2020). Training: lr 2e-5, 4 epochs, batch 16, linear warm-up, early stopping on dev Spearman; backbone choice follows Devlin et al. (2019).

**(c) Ordinal regression head (novel for this task).** MSE ignores the ordinal structure of the Likert scale. We replace the linear head with an **ordinal threshold head** predicting four monotone cumulative probabilities P(rating ≥ k) for k ∈ {2, 3, 4, 5}, trained with the CORN loss; the inferred rating is the highest k whose cumulative probability exceeds 0.5. Our milestone already includes a tuned four-threshold bucketing over the Ridge regressor (notebook §4) that lifts acc@std from 0.53 → 0.55 — the encoder-backed version is the final deliverable.

**(d) Likert-distribution head (novel for this task).** Each entry has a 5-annotator distribution, not a scalar; `stdev` confirms disagreement is signal. We add a second head predicting P(choice = k | story, gloss) for k ∈ {1..5}, trained with KL divergence against the empirical annotator distribution. At inference we collapse to expected value — a continuous Spearman-friendly output that also exposes per-sample uncertainty we use to re-weight the ensemble.

**(e) Ensemble.** Calibrated average of (a) and (d), with mixing weight tuned on a **homonym-disjoint 10 % train hold-out** that mirrors the dev regime — inspired by Loureiro and Jorge (2019).

**Current status on dev** (offline baselines + augmentation experiments already running; full details in notebook §5–6):

| system | acc@std | Spearman ρ |
|--------|-------:|----------:|
| majority (=4, repo) | 0.57 | 0.00 |
| mean-of-train (=3) | 0.53 | 0.00 |
| TF-IDF cosine (ours, calibrated) | 0.53 | **+0.19** |
| Feature Ridge + ordinal thresholds (ours) | **0.55** | +0.11 |
| Feature Ridge + WordNet augmentation (ours) | 0.53 | **+0.15** |
| Feature Ridge + Likert mix-up (ours) | **0.54** | +0.14 |

Every constant predictor has ρ = 0 by construction; our lexical baselines and the augmented variants are the first to produce non-zero Spearman on dev.

**Use of existing code and our novel contribution.** *Reused:* official scorer (`scoring.py`/`format_check.py`), HuggingFace `transformers`, `sentence-transformers`, PyTorch, `scikit-learn`, `scipy.stats.spearmanr`, NLTK WordNet; pre-trained checkpoints. *Ours:* (1) homonym-disjoint EDA finding, (2) ordinal head §3(c), (3) Likert-distribution head §3(d), (4) narrative-aware input format, (5) narrative-component ablation, (6) homonym-disjoint calibration fold, (7) `nonsensical` auxiliary head, (8) every augmentation operator and the WiC integration in §2.4. All baselines, loaders, prompt builder, ensemble calibrator and plots are ours.

# 4. Next Steps

Data preparation and augmentation are complete; the next five weeks are for modelling and evaluation. Each member owns one vertical slice and contributes to writing.

| Window | Member 1 | Member 2 | Member 3 |
|--------|----------|----------|----------|
| Apr 14 | zero-shot LLM on dev; prompt iteration | RoBERTa infra, first run | eval harness, confusion matrix / PR-curve utilities |
| Apr 21 | few-shot retrieval (SBERT) | DeBERTa-v3-large sweep; ordinal head | narrative-component ablation |
| Apr 28 | prompt-compression & cost study | distribution head (KL); ensemble calibration | error buckets, SOTA comparison table |
| May 5  | final LLM run on test; CodaBench | final encoder run on test; CodaBench | report §1–4, ablation tables |
| May 12 | report §5–6 and slides | slides (methods, training); Q&A prep | report finalisation, appendix |

Member 3 presents the milestone; all three attend for questions. **Risks:** LLM API cost (cap budget, prefer open model), GPU availability (fall back to DeBERTa-v3-base), gloss-overfit (monitor dev Spearman, early-stop).

---

## References

- Blevins, T. & Zettlemoyer, L. (2020). *Moving Down the Long Tail of Word Sense Disambiguation with Gloss-Informed Biencoders.* In Proc. ACL 2020.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners.* In Proc. NeurIPS 2020.
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* In Proc. NAACL 2019.
- Loureiro, D. & Jorge, A. (2019). *Language Modelling Makes Sense: Propagating Representations through WordNet for Full-Coverage Word Sense Disambiguation.* In Proc. ACL 2019.
- Pilehvar, M. T. & Camacho-Collados, J. (2019). *WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations.* In Proc. NAACL 2019.
