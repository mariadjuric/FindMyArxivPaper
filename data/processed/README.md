# Processed data layout

This directory now supports both the original FMAP atlas pipeline and the new **FMAP-RAG Lab** direction.

## Planned artifacts

### Existing FMAP artifacts
- `papers_processed.csv`
- `embeddings.npy`

### New FMAP-RAG Lab artifacts
- `papers_metadata.parquet` — normalized paper-level metadata
- `paper_sections.parquet` — optional section-aware extracted full text
- `paper_chunks.parquet` — chunked retrieval units with metadata
- `chunk_embeddings.npy` or vector-store export
- `benchmark_splits/` — optional benchmark-specific paper subsets

## Chunk schema draft

Each retrieval chunk should eventually include:

- `paper_id`
- `chunk_id`
- `section_title`
- `chunk_text`
- `chunk_index`
- `page_start`
- `page_end`
- `published`
- `category`
- `title`
- `authors`
- `url`

The first week only defines this schema and project structure. Full-text extraction and chunk population come in the next milestone.
