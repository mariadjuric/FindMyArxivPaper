from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class ChunkingConfig:
    chunk_words: int = 160
    overlap_words: int = 40
    min_chunk_words: int = 40


def estimate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3))


def contains_equation_like_text(text: str) -> bool:
    patterns = [r"\$[^$]+\$", r"\\[a-zA-Z]+", r"\b[A-Za-z]_\{?[A-Za-z0-9]+\}?", r"="]
    return any(re.search(pattern, text) for pattern in patterns)


def contains_citation_marker(text: str) -> bool:
    patterns = [r"\[[0-9,; ]+\]", r"\([A-Z][A-Za-z]+ et al\.,? \d{4}\)", r"\([A-Z][A-Za-z]+,? \d{4}\)"]
    return any(re.search(pattern, text) for pattern in patterns)


def split_text_into_chunks(text: str, config: ChunkingConfig) -> list[tuple[int, int, str]]:
    words = text.split()
    if not words:
        return []

    chunks: list[tuple[int, int, str]] = []
    step = max(config.chunk_words - config.overlap_words, 1)
    for start in range(0, len(words), step):
        end = min(start + config.chunk_words, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < config.min_chunk_words and chunks:
            break
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
        if end >= len(words):
            break
    return chunks


def build_chunk_records(df: pd.DataFrame, config: ChunkingConfig | None = None) -> pd.DataFrame:
    config = config or ChunkingConfig()
    records: list[dict] = []

    for _, row in df.iterrows():
        paper_id = row.get("benchmark_paper_id") or row.get("paper_id")
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        combined = f"{title}. {abstract}".strip()
        chunks = split_text_into_chunks(combined, config)

        for idx, (start_word, end_word, chunk_text) in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx:03d}"
            records.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "section_title": "title_abstract",
                    "section_path": "title_abstract",
                    "page_start": None,
                    "page_end": None,
                    "char_start": None,
                    "char_end": None,
                    "title": title,
                    "authors": row.get("authors", ""),
                    "category": row.get("category", ""),
                    "published": row.get("published", ""),
                    "url": row.get("url", ""),
                    "arxiv_id": str(row.get("url", "")).rstrip("/").split("/")[-1].replace("v1", "").replace("v2", ""),
                    "topic_cluster": row.get("topic_cluster", "unassigned"),
                    "token_count": estimate_token_count(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "contains_citation_marker": contains_citation_marker(chunk_text),
                    "contains_equation_like_text": contains_equation_like_text(chunk_text),
                    "source_word_start": start_word,
                    "source_word_end": end_word,
                }
            )
    return pd.DataFrame(records)
