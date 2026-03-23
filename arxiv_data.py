from __future__ import annotations

import socket
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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


class ArxivFetchError(RuntimeError):
    pass


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
    timeout_seconds: int = 30,
    allow_cached_fallback: bool = True,
) -> pd.DataFrame:
    categories = categories or list(DEFAULT_ARXIV_CATEGORIES)
    search_query = build_category_query(categories)
    records: list[dict] = []
    start = 0
    total_needed = int(max_results)
    partial_failure: Exception | None = None

    while start < total_needed:
        current_batch = min(batch_size, total_needed - start)
        try:
            batch_records = _fetch_batch(
                search_query,
                start=start,
                max_results=current_batch,
                timeout_seconds=timeout_seconds,
            )
        except ArxivFetchError as exc:
            partial_failure = exc
            break

        records.extend(batch_records)
        start += current_batch
        if start < total_needed:
            time.sleep(3)

    if records:
        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    if allow_cached_fallback and output_path.exists():
        return pd.read_csv(output_path)

    if partial_failure is not None:
        raise ArxivFetchError(
            f"Unable to fetch arXiv data and no cached dataset was available at {output_path}. "
            f"Last error: {partial_failure}"
        ) from partial_failure

    raise ArxivFetchError(
        f"No arXiv records were fetched and no cached dataset was available at {output_path}."
    )


def _fetch_batch(search_query: str, start: int, max_results: int, timeout_seconds: int) -> list[dict]:
    url = f"{ARXIV_API_URL}?search_query={quote(search_query)}&start={int(start)}&max_results={int(max_results)}&sortBy=submittedDate&sortOrder=descending"
    request = Request(url, headers={"User-Agent": "FMAP/1.0 (FindMyArxivPaper physics atlas builder; respectful batched fetch)"})

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                xml_text = response.read().decode("utf-8")
            return _parse_feed(xml_text, offset=start)
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise ArxivFetchError(f"arXiv returned HTTP {exc.code} for batch start={start}, size={max_results}") from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise ArxivFetchError(f"Timed out reading arXiv response for batch start={start}, size={max_results}") from exc
        except URLError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise ArxivFetchError(f"Network error while fetching arXiv batch start={start}, size={max_results}: {exc}") from exc

    raise ArxivFetchError(
        f"Failed to fetch arXiv batch start={start}, size={max_results}. Last error: {last_error}"
    )


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
