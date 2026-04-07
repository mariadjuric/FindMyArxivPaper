from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .chunking import ChunkingConfig, contains_citation_marker, contains_equation_like_text, estimate_token_count, split_text_into_chunks


@dataclass
class SectionChunkingConfig(ChunkingConfig):
    include_title_prefix: bool = True


def build_section_chunks(fulltext_df: pd.DataFrame, manifest_df: pd.DataFrame, config: SectionChunkingConfig | None = None) -> pd.DataFrame:
    config = config or SectionChunkingConfig()
    manifest_lookup = manifest_df.set_index("benchmark_paper_id").to_dict(orient="index")
    records: list[dict] = []

    for _, row in fulltext_df.iterrows():
        paper_id = str(row["benchmark_paper_id"])
        meta = manifest_lookup.get(paper_id, {})
        sections = row.get("sections", []) or []
        chunk_counter = 0
        for section in sections:
            section_title = str(section.get("section_title", "full_text")).strip() or "full_text"
            section_text = str(section.get("section_text", "")).strip()
            if not section_text:
                continue
            if config.include_title_prefix:
                section_text = f"{meta.get('title', '')}\n\n{section_title}\n{section_text}".strip()
            for start_word, end_word, chunk_text in split_text_into_chunks(section_text, config):
                chunk_id = f"{paper_id}_chunk_{chunk_counter:03d}"
                chunk_counter += 1
                records.append(
                    {
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_counter - 1,
                        "chunk_text": chunk_text,
                        "section_title": section_title,
                        "section_path": section.get("section_path", section_title),
                        "page_start": None,
                        "page_end": None,
                        "char_start": None,
                        "char_end": None,
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                        "category": meta.get("category", ""),
                        "published": meta.get("published", ""),
                        "url": meta.get("url", ""),
                        "arxiv_id": str(meta.get("url", "")).rstrip('/').split('/')[-1].replace('v1', '').replace('v2', ''),
                        "topic_cluster": meta.get("topic_cluster", "unassigned"),
                        "token_count": estimate_token_count(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "contains_citation_marker": contains_citation_marker(chunk_text),
                        "contains_equation_like_text": contains_equation_like_text(chunk_text),
                        "source_word_start": start_word,
                        "source_word_end": end_word,
                    }
                )
    return pd.DataFrame(records)
