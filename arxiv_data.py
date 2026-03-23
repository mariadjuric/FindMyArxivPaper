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
    ASTRO_ONLY_PREFIX,
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


def build_year_query(from_year: int | None = None, to_year: int | None = None) -> str:
    if from_year is None and to_year is None:
        return ""
    start_year = from_year if from_year is not None else to_year
    end_year = to_year if to_year is not None else from_year
    if start_year is None or end_year is None:
        return ""
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    start_stamp = f"{start_year}0101000000"
    end_stamp = f"{end_year}1231235959"
    return f"submittedDate:[{start_stamp} TO {end_stamp}]"


def combine_query_parts(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return " AND ".join(f"({part})" for part in cleaned)


def fetch_arxiv_dataset(
    categories: list[str] | None = None,
    max_results: int = 500,
    output_path: Path = ARXIV_DATA_PATH,
    batch_size: int = 100,
    timeout_seconds: int = 30,
    allow_cached_fallback: bool = True,
    from_year: int | None = None,
    to_year: int | None = None,
) -> pd.DataFrame:
    categories = categories or list(DEFAULT_ARXIV_CATEGORIES)
    category_query = build_category_query(categories)
    target_count = int(max_results)
    records: list[dict] = []
    seen_urls: set[str] = set()
    partial_failure: Exception | None = None
    total_raw_seen = 0
    total_duplicates_skipped = 0
    total_non_astro_skipped = 0
    total_batches = 0

    if from_year is not None and to_year is not None and from_year > to_year:
        from_year, to_year = to_year, from_year

    year_ranges = _resolve_year_ranges(from_year=from_year, to_year=to_year)

    print(f"Target astro-ph papers: {target_count}")
    if year_ranges:
        print(f"Year filter: {year_ranges[0][0]} to {year_ranges[-1][1]}")
    else:
        print("Year filter: none (recent-tail query)")

    for window_start, window_end in year_ranges or [(None, None)]:
        if len(records) >= target_count:
            break
        window_query = combine_query_parts(category_query, build_year_query(window_start, window_end))
        window_label = f"{window_start}" if window_start == window_end else f"{window_start}-{window_end}" if window_start is not None else "all years"
        print(f"\nFetching window: {window_label}")

        start = 0
        exhausted = False
        window_batch = 0

        while len(records) < target_count and not exhausted:
            current_batch = min(batch_size, max(1, target_count - len(records)))
            window_batch += 1
            total_batches += 1
            try:
                batch_records = _fetch_batch(
                    window_query,
                    start=start,
                    max_results=current_batch,
                    timeout_seconds=timeout_seconds,
                )
            except ArxivFetchError as exc:
                partial_failure = exc
                exhausted = True
                break

            if not batch_records:
                print(f"Window {window_label} batch {window_batch}: arXiv returned 0 rows at start={start}. No more results available.")
                exhausted = True
                break

            start += current_batch
            total_raw_seen += len(batch_records)

            filtered_batch, skipped_non_astro, skipped_duplicates = _filter_and_dedupe_records(batch_records, seen_urls)
            total_non_astro_skipped += skipped_non_astro
            total_duplicates_skipped += skipped_duplicates
            records.extend(filtered_batch)

            print(
                f"Window {window_label} batch {window_batch}: fetched {len(batch_records)} raw | "
                f"kept {len(filtered_batch)} astro-ph | non-astro skipped {skipped_non_astro} | "
                f"duplicates skipped {skipped_duplicates} | total kept {len(records)}/{target_count}"
            )

            if len(batch_records) < current_batch:
                print(
                    f"Window {window_label} batch {window_batch}: arXiv returned fewer rows than requested "
                    f"({len(batch_records)} < {current_batch})."
                )
                exhausted = True
                break

            if len(records) < target_count and not exhausted:
                time.sleep(3)

    if records:
        df = _finalize_dataframe(pd.DataFrame(records), limit=target_count)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if len(df) < target_count:
            print(
                f"Requested {target_count} astro-ph papers, but only {len(df)} were available from arXiv for this query range. "
                "Using the maximum available."
            )
        else:
            print(
                f"Completed arXiv fetch with {len(df)} astro-ph papers retained from {total_raw_seen} raw results across {total_batches} batches."
            )
        if total_non_astro_skipped:
            print(f"Total non-astro rows skipped: {total_non_astro_skipped}")
        if total_duplicates_skipped:
            print(f"Total duplicates skipped: {total_duplicates_skipped}")
        return df

    if allow_cached_fallback and output_path.exists():
        cached = pd.read_csv(output_path)
        return _finalize_dataframe(cached, limit=target_count)

    if partial_failure is not None:
        raise ArxivFetchError(
            f"Unable to fetch arXiv data and no cached dataset was available at {output_path}. "
            f"Last error: {partial_failure}"
        ) from partial_failure

    raise ArxivFetchError(
        f"No arXiv records were fetched and no cached dataset was available at {output_path}."
    )


def _resolve_year_ranges(from_year: int | None, to_year: int | None) -> list[tuple[int, int]]:
    if from_year is None and to_year is None:
        return []
    start_year = from_year if from_year is not None else to_year
    end_year = to_year if to_year is not None else from_year
    if start_year is None or end_year is None:
        return []
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    return [(year, year) for year in range(end_year, start_year - 1, -1)]


def _filter_and_dedupe_records(records: list[dict], seen_urls: set[str]) -> tuple[list[dict], int, int]:
    kept: list[dict] = []
    skipped_non_astro = 0
    skipped_duplicates = 0
    for record in records:
        label = str(record.get(LABEL_COLUMN, ""))
        url = str(record.get(URL_COLUMN, "")).strip()
        if not label.startswith(ASTRO_ONLY_PREFIX):
            skipped_non_astro += 1
            continue
        dedupe_key = url or f"{record.get(TITLE_COLUMN, '')}::{record.get(PUBLISHED_COLUMN, '')}"
        if dedupe_key in seen_urls:
            skipped_duplicates += 1
            continue
        seen_urls.add(dedupe_key)
        kept.append(record)
    return kept, skipped_non_astro, skipped_duplicates


def _finalize_dataframe(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if LABEL_COLUMN in df.columns:
        df = df[df[LABEL_COLUMN].astype(str).str.startswith(ASTRO_ONLY_PREFIX)].reset_index(drop=True)
    if URL_COLUMN in df.columns:
        df = df.drop_duplicates(subset=[URL_COLUMN], keep="first")
    if limit is not None:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    df[ID_COLUMN] = range(1, len(df) + 1)
    return df


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
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        primary = primary.attrib.get("term", "") if primary is not None else ""
        if not primary:
            astro_categories = [cat for cat in categories_for_entry if cat.startswith(ASTRO_ONLY_PREFIX)]
            primary = astro_categories[0] if astro_categories else (categories_for_entry[0] if categories_for_entry else "unknown")

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
