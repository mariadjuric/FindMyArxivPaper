# SciPaper

**SciPaper** is a compact research-style NLP project for scientific paper classification, semantic retrieval, and embedding visualization.

This repository is intentionally built as a clean portfolio-style ML pipeline:
- load paper metadata from CSV
- preprocess text
- build dense embeddings for retrieval and plots
- train a stronger supervised text classifier
- evaluate with accuracy, macro F1, confusion matrix, and retrieval precision@k
- save figures and model artifacts

There is no product-name ambiguity here: the project name is **SciPaper**.

---

## What the project is doing

SciPaper works on paper metadata with at least these columns:
- `title`
- `abstract`
- `category`

The pipeline uses the data in two different ways:

### 1. Retrieval and visualization path
For semantic search and 2D embedding plots, the project builds sentence-transformer embeddings from:
- `title + abstract`

Model used:
- `sentence-transformers/all-MiniLM-L6-v2`

Those embeddings are used for:
- semantic search
- retrieval evaluation
- PCA/t-SNE style visual exploration

### 2. Supervised classification path
For category prediction, the project uses a stronger classic text baseline over:
- `title + abstract`

Classifier method:
- **TF-IDF vectorization** with unigram + bigram features
- **LinearSVC** classifier

So the current classifier is **not** embedding -> logistic regression anymore.
It is now a more competitive text classification baseline that usually behaves better on small and medium tabular text datasets.

---

## Synthetic datasets included

The repo now includes two synthetic CSV datasets in `data/raw/`:

### `papers.csv`
The main synthetic dataset.

- about **1000 papers**
- balanced across multiple scientific categories
- intentionally larger than the original toy sample
- designed to be useful for testing the pipeline without being completely trivial
- includes some shared language across classes so it is less unrealistically clean

### `papers_perfect.csv`
A second synthetic dataset for debugging/demo purposes.

- also about **1000 papers**
- intentionally **too separable**
- category wording is highly distinctive
- useful when you want a near-perfect confusion matrix or to verify that the pipeline works end to end

In short:
- use `papers.csv` for a more believable demo
- use `papers_perfect.csv` when you want a stress-free synthetic sanity check

By default, the pipeline reads:
- `data/raw/papers.csv`

If you want to try the perfect synthetic set, replace `papers.csv` with `papers_perfect.csv` or adjust the configured input path.

---

## Why this is a good portfolio project

This project demonstrates:
- **NLP representation learning** with sentence-transformer embeddings
- **text classification** with a strong linear baseline
- **semantic retrieval** over scientific abstracts
- **evaluation discipline** with saved metrics and plots
- **research engineering** through modular code and reproducible outputs

That makes it a much stronger repo than a one-off notebook or a generic chatbot demo.

---

## Project structure

```text
SciPaper/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── models.py
├── train.py
├── evaluate.py
├── plots.py
├── search.py
├── data.py
├── utils.py
├── demo.py
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── papers.csv
│   │   └── papers_perfect.csv
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── notebooks/
    └── exploration.ipynb
```

---

## Pipeline overview

### 1. Data loading and preprocessing
`data.py` loads the CSV, validates required columns, cleans missing text, and creates:
- `combined_text = title + ". " + abstract`

That combined text field is used by both the embedding pipeline and the classifier.

### 2. Embedding model
`models.py` wraps a sentence-transformer encoder.

Embeddings are used for:
- semantic search
- retrieval scoring
- embedding projection plots

Default embedding model:
- `sentence-transformers/all-MiniLM-L6-v2`

### 3. Supervised classifier
`train.py` fits the classifier on `combined_text`.

Current baseline:
- `TfidfVectorizer(ngram_range=(1, 2), max_features=8000)`
- `LinearSVC`

This is a strong and very common baseline for document classification.
It is fast, simple, and often surprisingly competitive.

### 4. Evaluation
`evaluate.py` computes:
- accuracy
- macro F1
- confusion matrix
- classification report
- retrieval precision@k

### 5. Visualization
`plots.py` generates:
- label distribution plots
- 2D embedding projection plots
- confusion matrix heatmap

---

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- load `data/raw/papers.csv`
- preprocess title + abstract text
- build embeddings
- train the TF-IDF + LinearSVC classifier
- evaluate metrics
- save figures and outputs
- print example semantic search results

### 3. Run the demo search

```bash
python demo.py --query "graph neural networks for molecules" --top_k 5
```

---

## Input data format

Expected CSV format:

```csv
title,abstract,category,year,paper_id
Paper A,"This paper studies transformers for vision.",cs.CV,2024,1
Paper B,"We present a model for scientific retrieval.",cs.CL,2023,2
```

Default input path:

```text
data/raw/papers.csv
```

---

## Main files explained

### `data.py`
Handles:
- synthetic dataset generation
- CSV loading
- preprocessing
- train/test splitting

### `models.py`
Contains:
- `PaperEmbedder` for sentence-transformer embeddings
- `PaperClassifier` for TF-IDF + LinearSVC classification

### `train.py`
Handles:
- fitting the classifier
- saving trained model artifacts

### `evaluate.py`
Handles:
- computing metrics
- writing JSON results
- retrieval evaluation

### `main.py`
Runs the whole workflow:
1. load data
2. preprocess text
3. build embeddings
4. train classifier
5. evaluate results
6. generate plots
7. run example retrieval queries

### `plots.py`
Creates the key visual outputs for inspection and reporting.

---

## Outputs

After running `python main.py`, the main artifacts are written to:

- `outputs/models/` — trained classifier
- `outputs/metrics/` — JSON metrics
- `outputs/figures/` — plots and confusion matrix

---

## Notes on scope

SciPaper is intentionally a strong baseline repo, not a huge framework.

The goal is:
- easy to run
- easy to understand
- easy to extend with better data or richer models

That makes it a solid public GitHub or portfolio project.
