# FMAP v2: Comparing the New Transformer Classifier Against the v1 Baseline on astro-ph arXiv Data

When I first built **FMAP: FindMyArxivPaper**, the classification layer was deliberately simple: a **TF-IDF + LinearSVC** baseline over paper titles and abstracts. That was the right first move. It was fast, interpretable, and strong enough to prove the project idea.

But the project has grown into something more serious: real **astro-ph arXiv ingestion**, a larger interactive paper atlas, semantic retrieval, and now a new **v2 transformer classifier** fine-tuned directly on astrophysics papers.

This post looks at what changed in **v2**, what stayed the same, and how I want to compare it against **v1** on the same astro-ph dataset coming from the arXiv ingestion pipeline.

---

## The setup

For this comparison, the goal is not to test on synthetic data or hand-separated toy examples.

It is to compare **v1 vs v2 on real astro-ph papers** fetched through the FMAP ingestion path.

### Dataset

The current arXiv CSV used by FMAP contains **5,000 astro-ph papers** retained from the export API, with these classes:

- `astro-ph.CO`
- `astro-ph.EP`
- `astro-ph.GA`
- `astro-ph.HE`
- `astro-ph.IM`
- `astro-ph.SR`

Current class counts in the retained CSV:

- `astro-ph.GA`: 1268
- `astro-ph.HE`: 1059
- `astro-ph.SR`: 809
- `astro-ph.CO`: 740
- `astro-ph.EP`: 626
- `astro-ph.IM`: 498

FMAP uses a **75/25 train/test split** with stratification when possible, so this comparison is done on the same split logic used elsewhere in the project.

### Input text

Both models use the same text field:

- **title + abstract**

That matters because it keeps the comparison fair. The difference is in the modeling approach, not in the information given to the model.

---

## v1: the baseline

The original classifier is intentionally classical ML:

- **Vectorizer:** TF-IDF
- **Features:** unigrams + bigrams
- **Classifier:** `LinearSVC`

Why this baseline still matters:

- it is very fast to train
- it is cheap to rerun when iterating on datasets
- it gives a clean reference point for future work
- it is surprisingly competitive for scientific text classification

This is the kind of model you keep around even after adding fancier models, because otherwise you stop knowing whether the extra complexity is actually buying you anything.

---

## v2: the new classifier

The new **v2** path moves classification into transformer fine-tuning.

### Architecture

- **Framework:** PyTorch + Hugging Face Transformers
- **Default backbone:** `allenai/scibert_scivocab_uncased`
- **Training mode:** supervised fine-tuning
- **Task:** 6-way astro-ph category classification

SciBERT is a natural fit here because FMAP is not classifying tweets or product reviews. It is working on scientific titles and abstracts, where domain vocabulary and phrase context matter.

### What changed technically

Compared with v1, v2 adds:

- a transformer-based classifier wrapper
- tokenizer/model checkpoint saving
- versioned CLI training flow
- experiment controls for epochs, batch size, sequence length, learning rate, and backbone choice
- a cleaner separation between the **classifier** and the **embedding/retrieval** side of the project

In FMAP, that split is now explicit:

- **classifier** = v1 baseline or v2 transformer
- **retrieval / paper map** = sentence embeddings (`all-MiniLM-L6-v2`)

That is an important design decision. The 2D atlas and semantic search do not depend on the classification model being transformer-based.

---

## Current v2 result on astro-ph arXiv data

The current saved v2 evaluation on the 5,000-paper astro-ph dataset is:

- **Accuracy:** `0.8824`
- **Macro F1:** `0.8778`
- **Model:** `transformer_finetune:allenai/scibert_scivocab_uncased`

Per-class F1 scores:

- `astro-ph.CO`: **0.9160**
- `astro-ph.EP`: **0.9068**
- `astro-ph.GA`: **0.8946**
- `astro-ph.HE`: **0.8818**
- `astro-ph.IM`: **0.8015**
- `astro-ph.SR`: **0.8660**

A few immediate observations:

1. **The model is strong overall.** An 0.878 macro F1 on six related astro-ph subclasses is a solid result.
2. **Instrumentation (`astro-ph.IM`) is the hardest class** in the current run. That is not too surprising: instrumentation often overlaps linguistically with neighboring observational or analysis-heavy work.
3. **Cosmology (`astro-ph.CO`) and Earth/planetary (`astro-ph.EP`) are particularly strong** in the current checkpoint.
4. The result is balanced enough that it does not look like the model is only coasting on the largest class.

---

## Why v2 should be better in principle

Even before finishing the exact side-by-side v1 rerun, the reason to expect v2 improvements is fairly clear.

### 1. Context matters in scientific abstracts

TF-IDF is good at weighting terms, but it does not understand phrase context in the transformer sense. Scientific abstracts often distinguish subfields through combinations of ideas, not just isolated keywords.

For example, a transformer has a much better shot at separating:

- observational instrumentation language
n- galaxy-formation context
- cosmological inference phrasing
- high-energy phenomenology terms

when the differences live in how those terms are composed, not just whether they appear.

### 2. SciBERT is domain-matched

Using a scientific-text backbone matters. The model starts from a representation space that already knows something about the syntax and vocabulary of research writing.

### 3. v2 is a better platform for future experiments

Even if the raw improvement over v1 were modest, v2 gives FMAP a more serious experimental path:

- stronger backbones
- class weighting
- better calibration
- retrieval-aware objectives
- contrastive or multitask extensions

v1 is a baseline. v2 is a research direction.

---

## What I want the comparison section to show

A good portfolio comparison should be more than “new model good.” I want it to answer four things clearly.

### 1. What is different?

**v1**
- TF-IDF features
- linear classifier
- fast, light, interpretable

**v2**
- transformer fine-tuning
- contextual encoding
- slower, heavier, more expressive

### 2. What is better?

The “better” part should focus on:

- overall accuracy / macro F1
- per-class improvements
- whether minority or ambiguous classes benefit
- whether the confusion matrix becomes more sensible

### 3. What are the tradeoffs?

It would be a bit silly to pretend v2 is strictly better in every practical sense.

Tradeoffs are real:

- **v1 wins on speed**
- **v1 wins on simplicity**
- **v2 likely wins on classification quality**
- **v2 wins on project sophistication / realism**

### 4. Why does this matter for FMAP as a project?

Because FMAP is not just a notebook experiment anymore. It now combines:

- real arXiv ingestion
- supervised paper classification
- semantic retrieval
- interactive atlas visualization
- historical paper collection over astro-ph

That makes the v2 upgrade meaningful in the overall project story. It shows FMAP evolving from a nice baseline pipeline into a more serious NLP system built on real scientific data.

---

## A clean comparison table

This is the structure I want in the final published post once the v1 rerun is pinned down from the same CSV and split:

| Model | Text representation | Classifier | Dataset | Accuracy | Macro F1 | Notes |
|---|---|---|---|---:|---:|---|
| v1 | TF-IDF (1-2 grams) | LinearSVC | 5k astro-ph arXiv papers | _TBD rerun_ | _TBD rerun_ | fast baseline |
| v2 | SciBERT fine-tuning | Transformer classifier | 5k astro-ph arXiv papers | 0.8824 | 0.8778 | stronger contextual model |

I would also add:

- confusion matrices for both
- a short per-class delta table
- a paragraph on runtime/training-cost differences

---

## The main takeaway

The important story is not just that FMAP has a new classifier.

It is that the project now has **two distinct modeling tiers**:

- a strong lightweight baseline that is easy to trust and rerun
- a more expressive transformer model that is better aligned with scientific language and future experimentation

That is the kind of comparison I like in portfolio work. Not complexity for its own sake, but a clearer ladder of methods:

1. ingest real data
2. build a solid baseline
3. add a more serious model
4. compare them honestly
5. show where the extra complexity is worth it

That is exactly what FMAP v2 is trying to do.

---

## Next step

The next step is simple:

- rerun **v1** on the exact same astro-ph CSV used for the saved v2 result
- export both metric files side by side
- add the finished comparison table and confusion-matrix figures to the post

Once that is done, this becomes a proper “v1 vs v2 on real arXiv astro-ph classification” write-up rather than just an architectural update.

And honestly, that is the better story anyway.
