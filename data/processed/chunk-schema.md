# FMAP-RAG Lab chunk schema

This file defines the concrete retrieval-unit schema for the first implementation phase.

The chunk is the main evidence object for retrieval, reranking, citation, and factuality evaluation.

---

## Chunk record schema

Each chunk record should contain the following fields.

### Identity
- `paper_id` — stable paper identifier
- `chunk_id` — stable chunk identifier
- `chunk_index` — integer chunk order within the paper

### Text content
- `chunk_text` — chunk body text
- `section_title` — section heading if available
- `section_path` — optional hierarchical section path, e.g. `Methods > Data`

### Source localization
- `page_start` — first page index if available
- `page_end` — last page index if available
- `char_start` — character offset in source text if available
- `char_end` — character offset in source text if available

### Paper metadata
- `title` — paper title
- `authors` — author string
- `category` — astro-ph category or local topic label
- `published` — publication timestamp
- `url` — arXiv or source URL
- `arxiv_id` — arXiv identifier if available
- `topic_cluster` — benchmark cluster label

### Retrieval metadata
- `token_count` — estimated token count
- `word_count` — word count
- `contains_citation_marker` — boolean
- `contains_equation_like_text` — boolean

---

## Design goals

The chunk schema should support:
- dense retrieval
- lexical retrieval
- reranking
- answer citation rendering
- manual evidence inspection
- claim-level support checking

That means chunk ids must be stable and human-readable enough to inspect during debugging.

---

## Recommended chunk id format

Use:

```text
{paper_id}_chunk_{chunk_index:03d}
```

Example:

```text
galdyn_001_chunk_007
```

---

## First implementation policy

For the first retrieval baseline:
- use fixed-size text chunks with overlap
- preserve section titles where possible
- keep the schema stable even if some fields are null at first

Do **not** wait for perfect PDF parsing before freezing the schema. Stable identifiers and metadata discipline matter more than perfect extraction in the first pass.
