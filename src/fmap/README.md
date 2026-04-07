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
