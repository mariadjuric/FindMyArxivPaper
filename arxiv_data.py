from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import time
import xml.etree.ElementTree as ET

import pandas as pd

from config import (
    ARXIV_DATA_PATH,
    AUTHORS_COLUMN,
    DEFAULT_ARXIV_CATEGORIES,
    ID_COLUMN,
    LABEL_COLUMN,
    PRIMARY_CATEGORY_COLUMN,
    PUBLISHED_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    UPDATED_COLUMN,
    URL_COLUMN,
    YEAR_COLUMN,
)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_category_query(categories: Iterable[str]) -> str:
    cats = [c.strip() for c in categories if c.strip()]
    if not cats:
        cats = list(DEFAULT_ARXIV_CATEGORIES)
    return " OR ".join(f"cat:{c}" for c in cats)


def fetch_arxiv_dataset(
    categories: list[str] | None = None,
    max_results: int = 500,
    output_path: Path = ARXIV_DATA_PATH,
    batch_size: int = 100,
) -> pd.DataFrame:
    categories = categories or list(DEFAULT_ARXIV_CATEGORIES)
    search_query = build_category_query(categories)
    records: list[dict] = []
    start = 0
    total_needed = int(max_results)

    while start < total_needed:
        current_batch = min(batch_size, total_needed - start)
        records.extend(_fetch_batch(search_query, start=start, max_results=current_batch))
        start += current_batch
        if start < total_needed:
            time.sleep(3)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _fetch_batch(search_query: str, start: int, max_results: int) -> list[dict]:
    url = f"{ARXIV_API_URL}?search_query={quote(search_query)}&start={int(start)}&max_results={int(max_results)}&sortBy=submittedDate&sortOrder=descending"
    request = Request(url, headers={"User-Agent": "SciPaper/1.0 (physics atlas builder; contact via local run)"})

    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                xml_text = response.read().decode("utf-8")
            return _parse_feed(xml_text, offset=start)
        except HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            if exc.code == 429 and ARXIV_DATA_PATH.exists():
                return []
            raise

    return []


def _parse_feed(xml_text: str, offset: int = 0) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ATOM_NS)
    records: list[dict] = []

    for idx, entry in enumerate(entries, start=1 + offset):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        updated = entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
        link = entry.findtext("atom:id", default="", namespaces=ATOM_NS)

        author_names = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories_for_entry = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ATOM_NS)]
        primary = categories_for_entry[0] if categories_for_entry else "unknown"

        records.append(
            {
                ID_COLUMN: idx,
                TITLE_COLUMN: title,
                TEXT_COLUMN: abstract,
                LABEL_COLUMN: primary,
                PRIMARY_CATEGORY_COLUMN: primary,
                AUTHORS_COLUMN: ", ".join(a for a in author_names if a),
                PUBLISHED_COLUMN: published,
                UPDATED_COLUMN: updated,
                URL_COLUMN: link,
                YEAR_COLUMN: published[:4] if published else "",
            }
        )

    return records


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())
