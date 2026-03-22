# SciPaper

**SciPaper** is a research-style NLP project for exploring scientific literature with modern representation learning, retrieval, clustering, and lightweight summarization.

If you want a cleaner product name later, good alternatives are:
- **LitScope**
- **PaperScope**
- **ArxivAtlas**
- **LitGraph**

For now, `SciPaper` is a clear repo name and easy to understand.

---

## What this project does

SciPaper builds an end-to-end pipeline over scientific paper metadata and abstracts.

Given a collection of papers, it can:
- load and clean data
- create train/dev/test splits
- encode abstracts into dense embeddings
- train a lightweight paper-domain classifier baseline
- build semantic search over embeddings
- cluster papers into topics
- evaluate retrieval and classification quality
- generate plots for exploration and reporting

This repo is designed to look like a small ML research project rather than a one-off demo.

---

## Why this is a good portfolio project

This project demonstrates:
- **NLP representation learning** with transformer embeddings
- **information retrieval** with semantic similarity search
- **classification** for a reproducible supervised baseline
- **unsupervised learning** through clustering and visualization
- **evaluation discipline** through explicit metrics and plots
- **research engineering** through modular code, configs, scripts, and outputs

For ML research applications, that combination is much stronger than a generic chatbot demo.

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

### 1. Data loading
`data.py` loads a CSV containing paper metadata. At minimum, the CSV should contain:
- `title`
- `abstract`

Optional columns:
- `category`
- `year`
- `authors`
- `paper_id`

A small synthetic sample dataset is included so the pipeline runs immediately.

---

### 2. Embedding model
`models.py` wraps a sentence-transformer encoder. It converts each paper abstract into a dense vector.

Why use embeddings?
- Papers with similar meaning should be close in vector space.
- That makes semantic search and clustering possible.
- It is a standard building block in modern NLP systems.

Default model:
- `sentence-transformers/all-MiniLM-L6-v2`

This is lightweight, fast, and good for a public starter repo.

---

### 3. Supervised baseline
`train.py` trains a simple classifier on top of embeddings to predict paper category.

This is useful because it shows:
- a reproducible training pipeline
- a supervised benchmark
- evaluation beyond just retrieval

The baseline is intentionally simple:
- abstract -> embedding -> logistic regression classifier

That keeps the project fast and understandable while still demonstrating ML workflow.

---

### 4. Semantic retrieval
`search.py` computes nearest neighbors in embedding space.

Given a query string or paper abstract, the system:
1. embeds the query
2. compares it to all paper embeddings
3. returns the most similar papers

This forms the core of the literature explorer.

---

### 5. Clustering and visualization
`plots.py` reduces embeddings with PCA or t-SNE and plots 2D projections.

This helps answer questions like:
- Do papers naturally cluster by topic?
- Are some categories overlapping?
- Does the embedding space capture meaningful structure?

---

### 6. Evaluation
`evaluate.py` computes:
- classification accuracy
- macro F1
- confusion matrix
- retrieval precision@k

This is important because a good portfolio project should not stop at “it runs.” It should show how performance is measured.

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
- create/load a sample dataset
- encode abstracts
- train a classifier
- evaluate metrics
- save figures and outputs
- print example semantic search results

### 3. Run the lightweight demo search

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

Place your real dataset at:

```text
data/raw/papers.csv
```

If no real data is found, the repo falls back to a built-in sample dataset.

---

## Main files explained

### `models.py`
Contains model wrappers:
- `PaperEmbedder`: sentence-transformer embedding model
- `PaperClassifier`: logistic regression baseline over embeddings

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
The orchestration entry point. It runs the whole pipeline in the right order:
1. load data
2. clean and split
3. build embeddings
4. train classifier
5. evaluate
6. generate plots
7. run example retrieval queries

### `plots.py`
Creates:
- label distribution plots
- 2D embedding visualizations
- confusion matrix heatmap

---

## Outputs

After running `python main.py`, you should see artifacts in:

- `outputs/models/` — trained classifier and label encoder
- `outputs/metrics/` — JSON metrics files
- `outputs/figures/` — plots for README, portfolio, or reports

---

## Suggested next upgrades

If you want to turn this into a stronger research repo, the next steps are:

1. **Replace the sample data with arXiv metadata**
2. **Add a cross-encoder reranker** for better retrieval quality
3. **Add clustering with HDBSCAN or BERTopic**
4. **Add keyword extraction and paper summaries**
5. **Build a Streamlit front-end** for portfolio demos
6. **Compare multiple embedding models** and report results
7. **Add experiment tracking** with Weights & Biases or MLflow

---

## How to describe this on your portfolio

You can describe SciPaper like this:

> Built an end-to-end NLP system for scientific literature exploration using transformer embeddings, semantic retrieval, classification, and clustering. Designed modular training/evaluation pipelines, generated visual analyses of embedding structure, and created a reproducible codebase for experimenting with scientific document understanding.

That reads much better than “made an NLP app.”

---

## Notes on scope

This repo is intentionally a strong baseline, not a huge framework.

The idea is:
- easy to run
- easy to understand
- easy to expand into a more serious research project

That makes it ideal for a public GitHub repo and portfolio piece.
