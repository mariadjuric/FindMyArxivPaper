# FMAP: FindMyArxivPaper

**FMAP: FindMyArxivPaper** is a research-style project for exploring scientific papers with embeddings, classification, retrieval, and a generated interactive web atlas.

The project name is **FMAP: FindMyArxivPaper**.

It now supports two main modes:
- **synthetic demo mode** for quick testing
- **real arXiv mode** focused on astro-ph papers, with a generated interactive HTML visualization site

---

## What FMAP does now

FMAP can:
- ingest paper metadata from CSV
- fetch recent papers from the **arXiv API**
- focus on **astro-ph** categories by default
- build embeddings from **title + abstract**
- train a text classifier over categories
- evaluate retrieval and classification
- generate plots
- generate a **static interactive HTML site** with a coloured paper map

That makes it closer to a real paper-atlas project rather than just a toy CSV classifier.

---

## Real arXiv support

FMAP can fetch real arXiv metadata directly from the export API.

The fetcher is deliberately cautious:
- batched requests
- retry/backoff on rate limits and timeouts
- cached fallback to `data/raw/arxiv_astro_ph_papers.csv` if a previous fetch already succeeded

Default arXiv focus:
- `astro-ph.GA`
- `astro-ph.SR`
- `astro-ph.HE`
- `astro-ph.CO`
- `astro-ph.EP`
- `astro-ph.IM`

For each paper, FMAP stores fields such as:
- title
- abstract
- category
- authors
- published date
- arXiv URL

Fetched data is written to:
- `data/raw/arxiv_astro_ph_papers.csv`

---

## Interactive web atlas

FMAP now generates a static website at:
- `outputs/site/index.html`

The site includes:
- a dark themed atlas-style layout
- a **coloured astro-ph category map**
- interactive point selection
- search over title / abstract / author
- matched papers highlighted while non-matches are dimmed
- a details panel with category, date, abstract, authors, and arXiv link
- recommended nearby papers with approximate `% match` scores

The map is built from a 2D PCA projection of sentence-transformer embeddings over `title + abstract`.

This is a lightweight local/static visualization, so you can open the generated HTML directly in a browser.

---

## Modeling approach

FMAP uses the data in two different ways.

### 1. Embeddings for map + retrieval
Embedding model:
- `sentence-transformers/all-MiniLM-L6-v2`

Input text:
- `title + abstract`

Used for:
- semantic search
- retrieval evaluation
- 2D paper map projection
- interactive site visualization

### 2. Classifier for category prediction
Classifier method:
- **TF-IDF vectorization** with unigram + bigram features
- **LinearSVC**

Input text:
- `title + abstract`

This is the current supervised baseline. It is fast, strong, and easy to explain.

---

## Synthetic datasets

For demo/testing, FMAP still includes:
- `data/raw/papers.csv` — larger synthetic dataset
- `data/raw/papers_perfect.csv` — intentionally overly separable synthetic dataset

Use these when you want to test the pipeline without hitting arXiv.

---

## Project structure

```text
FindMyArxivPaper/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── data.py
├── arxiv_data.py
├── site_builder.py
├── models.py
├── train.py
├── evaluate.py
├── plots.py
├── search.py
├── demo.py
├── data/
│   ├── raw/
│   │   ├── papers.csv
│   │   ├── papers_perfect.csv
│   │   └── arxiv_astro_ph_papers.csv
│   └── processed/
└── outputs/
    ├── figures/
    ├── metrics/
    ├── models/
    └── site/
        ├── index.html
        └── data.js
```

---

## Quickstart

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run on synthetic data

```bash
python main.py --source synthetic
```

### 3. Run on the perfect synthetic dataset

```bash
python main.py --source perfect
```

### 4. Fetch real arXiv physics papers and build the site

```bash
python main.py --source arxiv --max-results 500
```

If arXiv is slow or rate-limits you, FMAP will retry and fall back to the cached CSV if one already exists.

### 5. Fetch custom arXiv categories

```bash
python main.py --source arxiv --max-results 800 --categories "astro-ph.GA,astro-ph.CO,astro-ph.HE,astro-ph.IM"
```

### 6. Open the generated website

After running the pipeline, open:

```text
outputs/site/index.html
```

---

## Example use cases

### Build a local physics paper atlas
Fetch astrophysics papers and generate a browsable map:

```bash
python main.py --source arxiv --max-results 600
open outputs/site/index.html
```

### Run semantic search on a chosen dataset

```bash
python demo.py --query "galaxy evolution and stellar populations" --input data/raw/arxiv_astro_ph_papers.csv
```

---

## Main files explained

### `arxiv_data.py`
Handles:
- arXiv API queries
- physics/astro category selection
- parsing Atom XML into a structured CSV

### `site_builder.py`
Handles:
- 2D projection normalization
- point/color payload generation
- writing the static HTML atlas

### `data.py`
Handles:
- synthetic dataset generation
- CSV loading
- preprocessing
- combined text creation
- train/test splitting

### `main.py`
Orchestrates the full workflow:
1. choose data source
2. optionally fetch arXiv data
3. preprocess text
4. build embeddings
5. train classifier
6. evaluate metrics
7. generate plots
8. generate the interactive site

---

## Outputs

After running the pipeline, you get:

- `outputs/models/` — trained classifier
- `outputs/metrics/` — evaluation metrics
- `outputs/figures/` — plots and confusion matrix
- `outputs/site/` — interactive HTML visualization site

---

## Why this is useful

This version of FMAP is much closer to a serious portfolio project because it combines:
- real data ingestion
- NLP embeddings
- text classification
- retrieval
- interactive visualization
- reproducible outputs

It is now a much clearer "physics paper atlas" style project while still staying lightweight and hackable.
