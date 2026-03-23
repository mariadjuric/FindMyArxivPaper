# SciPaper

**SciPaper** is a research-style project for exploring scientific papers with embeddings, classification, retrieval, and a generated interactive web atlas.

The project name stays **SciPaper**.

It now supports two main modes:
- **synthetic demo mode** for quick testing
- **real arXiv mode** focused on physics and astrophysics papers, with a generated HTML visualization site

---

## What SciPaper does now

SciPaper can:
- ingest paper metadata from CSV
- fetch recent papers from the **arXiv API**
- focus on **physics / astrophysics** categories by default
- build embeddings from **title + abstract**
- train a text classifier over categories
- evaluate retrieval and classification
- generate plots
- generate a **static interactive HTML site** with a coloured paper map

That makes it closer to a real paper-atlas project rather than just a toy CSV classifier.

---

## Real arXiv support

SciPaper can fetch real arXiv metadata directly from the export API.

Default arXiv focus:
- `astro-ph.GA`
- `astro-ph.SR`
- `astro-ph.HE`
- `astro-ph.CO`
- `astro-ph.EP`
- `astro-ph.IM`
- `physics.space-ph`
- `gr-qc`

For each paper, SciPaper stores fields such as:
- title
- abstract
- category
- authors
- published date
- arXiv URL

Fetched data is written to:
- `data/raw/arxiv_physics_papers.csv`

---

## Interactive web atlas

SciPaper now generates a static website at:
- `outputs/site/index.html`

The site includes:
- a dark themed atlas-style layout
- a **coloured category map**
- interactive point selection
- search over title / abstract / author
- a details panel with category, date, abstract, authors, and arXiv link

The map is built from a 2D PCA projection of sentence-transformer embeddings over `title + abstract`.

This is a lightweight local/static visualization, so you can open the generated HTML directly in a browser.

---

## Modeling approach

SciPaper uses the data in two different ways.

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

For demo/testing, SciPaper still includes:
- `data/raw/papers.csv` — larger synthetic dataset
- `data/raw/papers_perfect.csv` — intentionally overly separable synthetic dataset

Use these when you want to test the pipeline without hitting arXiv.

---

## Project structure

```text
SciPaper/
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
│   │   └── arxiv_physics_papers.csv
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

### 5. Fetch custom arXiv categories

```bash
python main.py --source arxiv --max-results 800 --categories "astro-ph.GA,astro-ph.CO,astro-ph.HE,gr-qc"
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
python demo.py --query "galaxy evolution and stellar populations" --input data/raw/arxiv_physics_papers.csv
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

This version of SciPaper is much closer to a serious portfolio project because it combines:
- real data ingestion
- NLP embeddings
- text classification
- retrieval
- interactive visualization
- reproducible outputs

It is now a much clearer "physics paper atlas" style project while still staying lightweight and hackable.
