# FMAP source layout

This directory is a lightweight scaffold for a cleaner source layout as FMAP grows into **FMAP-RAG Lab**.

## Intended modules

- `ingest/` — arXiv metadata fetch, later PDF/full-text acquisition
- `atlas/` — current interactive map generation and atlas-specific code
- `classify/` — v1/v2 category classification pipeline
- `retrieval/` — dense and lexical retrieval over paper chunks
- `rag/` — citation-grounded question answering
- `evaluation/` — retrieval, QA, citation, and factuality metrics
- `demo/` — local app / demonstration layer

The existing top-level scripts remain usable during the transition. This scaffold exists so the next refactor has a clear destination rather than drifting file-by-file.

## Current implementation status
The source tree now includes a first retrieval-oriented implementation slice:
- `benchmarks/corpus.py` — benchmark corpus manifest generation and topic-cluster assignment
- `benchmarks/link_questions.py` — provisional linking from benchmark questions to paper/chunk ids
- `retrieval/chunking.py` — stable chunk generation and chunk metadata
- `retrieval/baselines.py` — lexical and dense chunk retrieval baselines
- `evaluation/retrieval_eval.py` — Recall@k, MRR@k, and nDCG@k evaluation

This is still an early bridge layer. A first version of full PDF ingestion and section-aware extraction is now scaffolded, but reranking and citation-grounded answering are still to come.
