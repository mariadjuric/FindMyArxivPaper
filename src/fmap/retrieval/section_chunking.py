from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .chunking import ChunkingConfig, contains_citation_marker, contains_equation_like_text, estimate_token_count, split_text_into_chunks
from .section_filters import chunk_quality_score, is_preferred_section, is_probably_bad_section


@dataclass
class SectionChunkingConfig(ChunkingConfig):
    include_title_prefix: bool = True
    drop_bad_sections: bool = True
    prefer_semantic_sections: bool = True
    max_chunks_per_paper: int = 60
    max_chunks_per_section: int = 12
    drop_equation_heavy_chunks: bool = True
    drop_title_section: bool = True


def build_section_chunks(fulltext_df: pd.DataFrame, manifest_df: pd.DataFrame, config: SectionChunkingConfig | None = None) -> pd.DataFrame:
    config = config or SectionChunkingConfig()
    manifest_lookup = manifest_df.set_index("benchmark_paper_id").to_dict(orient="index")
    records: list[dict] = []

    for _, row in fulltext_df.iterrows():
        paper_id = str(row["benchmark_paper_id"])
        meta = manifest_lookup.get(paper_id, {})
        sections = row.get("sections", []) or []
        chunk_counter = 0
        paper_records: list[dict] = []
        for section in sections:
            section_title = str(section.get("section_title", "full_text")).strip() or "full_text"
            section_text = str(section.get("section_text", "")).strip()
            if not section_text:
                continue
            if config.drop_title_section and section_title.strip().lower() == str(meta.get('title', '')).strip().lower():
                continue
            if config.drop_bad_sections and is_probably_bad_section(section_title, section_text):
                continue
            formatted_text = section_text
            if config.include_title_prefix:
                formatted_text = f"{meta.get('title', '')}\n\n{section_title}\n{section_text}".strip()
            section_records: list[dict] = []
            for start_word, end_word, chunk_text in split_text_into_chunks(formatted_text, config):
                contains_eq = contains_equation_like_text(chunk_text)
                if config.drop_equation_heavy_chunks and contains_eq and not is_preferred_section(section_title):
                    continue
                record = {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}_chunk_{chunk_counter:03d}",
                    "chunk_index": chunk_counter,
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
                    "contains_equation_like_text": contains_eq,
                    "source_word_start": start_word,
                    "source_word_end": end_word,
                    "preferred_section": is_preferred_section(section_title),
                    "chunk_quality_score": chunk_quality_score(section_title, chunk_text),
                }
                chunk_counter += 1
                section_records.append(record)
            section_records.sort(key=lambda r: r["chunk_quality_score"], reverse=True)
            paper_records.extend(section_records[: config.max_chunks_per_section])

        if config.prefer_semantic_sections:
            paper_records.sort(key=lambda r: (r["preferred_section"], r["chunk_quality_score"]), reverse=True)
        paper_records = paper_records[: config.max_chunks_per_paper]
        for idx, record in enumerate(paper_records):
            record["chunk_index"] = idx
            record["chunk_id"] = f"{paper_id}_chunk_{idx:03d}"
        records.extend(paper_records)
    return pd.DataFrame(records)
