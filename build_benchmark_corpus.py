from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data import load_dataset
from src.fmap.benchmarks.corpus import build_corpus_manifest, save_corpus_manifest
from src.fmap.benchmarks.link_questions import attach_gold_targets
from src.fmap.retrieval.chunking import ChunkingConfig, build_chunk_records

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "raw" / "arxiv_astro_ph_papers.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "benchmark_corpus_manifest.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "paper_chunks.csv"
QUESTIONS_PATH = ROOT / "benchmarks" / "astrophysics_qa" / "questions.v0.json"
LINKED_QUESTIONS_PATH = ROOT / "benchmarks" / "astrophysics_qa" / "questions.linked.json"


def main() -> None:
    df = load_dataset(DATASET_PATH)
    manifest = build_corpus_manifest(df, limit=36)
    save_corpus_manifest(manifest, MANIFEST_PATH)

    chunks = build_chunk_records(manifest, ChunkingConfig(chunk_words=120, overlap_words=30, min_chunk_words=30))
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(CHUNKS_PATH, index=False)

    questions = pd.read_json(QUESTIONS_PATH)
    linked = attach_gold_targets(questions, chunks, manifest)
    with open(LINKED_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(linked.to_dict(orient="records"), f, indent=2)

    print(f"Saved manifest to {MANIFEST_PATH}")
    print(f"Saved chunks to {CHUNKS_PATH}")
    print(f"Saved linked questions to {LINKED_QUESTIONS_PATH}")
    print(f"Corpus papers: {len(manifest)} | chunks: {len(chunks)} | questions: {len(linked)}")


if __name__ == "__main__":
    main()
