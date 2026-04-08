from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.fmap.ingest.fulltext import FullTextIngestConfig, ingest_full_texts

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "processed" / "benchmark_corpus_manifest.csv"
FULLTEXT_CSV_PATH = ROOT / "data" / "processed" / "paper_fulltext.csv"
FULLTEXT_JSON_PATH = ROOT / "data" / "processed" / "paper_sections.json"
PDF_DIR = ROOT / "data" / "raw" / "pdfs"
EXTRACTED_DIR = ROOT / "data" / "processed" / "fulltext"


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    config = FullTextIngestConfig(pdf_dir=PDF_DIR, extracted_dir=EXTRACTED_DIR, overwrite=False, cache_format="md")
    fulltext = ingest_full_texts(manifest, config=config)

    serializable = fulltext.copy()
    FULLTEXT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable.drop(columns=["sections"], errors="ignore").to_csv(FULLTEXT_CSV_PATH, index=False)
    with open(FULLTEXT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(fulltext.to_dict(orient="records"), f, indent=2)

    counts = fulltext["status"].value_counts(dropna=False).to_dict()
    print(json.dumps({"status_counts": counts, "rows": len(fulltext)}, indent=2))


if __name__ == "__main__":
    main()
