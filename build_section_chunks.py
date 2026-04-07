from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.fmap.retrieval.section_chunking import SectionChunkingConfig, build_section_chunks

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "processed" / "benchmark_corpus_manifest.csv"
FULLTEXT_JSON_PATH = ROOT / "data" / "processed" / "paper_sections.json"
OUT_PATH = ROOT / "data" / "processed" / "paper_section_chunks.csv"


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    with open(FULLTEXT_JSON_PATH, "r", encoding="utf-8") as f:
        fulltext = pd.DataFrame(json.load(f))
    chunks = build_section_chunks(fulltext, manifest, SectionChunkingConfig(chunk_words=180, overlap_words=40, min_chunk_words=50))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(chunks)} section-aware chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
