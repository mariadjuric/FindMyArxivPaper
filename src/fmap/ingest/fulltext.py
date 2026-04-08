from __future__ import annotations

import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


@dataclass
class FullTextIngestConfig:
    pdf_dir: Path
    extracted_dir: Path
    request_pause_seconds: float = 1.0
    user_agent: str = "FMAP-RAG-Lab/0.1 (+local research project)"
    overwrite: bool = False


def arxiv_abs_to_pdf_url(url: str) -> str:
    if not url:
        return ""
    pdf_url = url.replace("/abs/", "/pdf/")
    if not pdf_url.endswith(".pdf"):
        pdf_url = f"{pdf_url}.pdf"
    return pdf_url


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def extract_arxiv_id(url: str) -> str:
    tail = (url or "").rstrip("/").split("/")[-1]
    return tail.replace(".pdf", "") or "unknown"


def download_pdf(pdf_url: str, dest_path: Path, user_agent: str) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return "cached"

    req = Request(pdf_url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as response:  # nosec - controlled HTTPS fetch for arXiv PDFs
        dest_path.write_bytes(response.read())
    return "downloaded"


def _normalize_line_for_repetition(line: str) -> str:
    line = re.sub(r"\s+", " ", (line or "").strip())
    line = re.sub(r"\b\d+\b", "#", line)
    return line.lower()


def _looks_like_running_header_or_footer(line: str) -> bool:
    raw = (line or "").strip()
    lower = raw.lower()
    if not raw:
        return False
    if re.fullmatch(r"\d+", raw):
        return True
    if "arxiv:" in lower:
        return True
    if any(token in lower for token in ["draft version", "typeset using", "compiled using", "preprint", "manuscript no."]):
        return True
    if re.search(r"\b(received|accepted|published|submitted)\b", lower):
        return True
    if re.fullmatch(r".*\b(page|vol\.?|volume|no\.?|issue)\b.*", lower):
        return True
    return False


def _repair_broken_words(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z]\b)", "", text)
    text = re.sub(r"(?<=[a-z])\s+(?=[a-z]{1,2}\b)", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[–—-][A-Za-z])", "", text)
    return text


def _clean_page_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if _looks_like_running_header_or_footer(line):
            continue
        line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
        line = re.sub(r"/hyphen\.alt", "-", line)
        line = re.sub(r"\s*([,.;:!?])", r"\1", line)
        line = re.sub(r"(?<=\w)-\s*$", "", line)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _repair_broken_words(text)
    return text.strip()


def _strip_repeated_margin_lines(page_texts: list[str]) -> list[str]:
    if len(page_texts) < 2:
        return page_texts

    candidate_counts: Counter[str] = Counter()
    page_candidates: list[set[str]] = []
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        edge_lines = lines[:3] + lines[-3:]
        normalized = {_normalize_line_for_repetition(line) for line in edge_lines if len(line.split()) >= 2}
        page_candidates.append(normalized)
        candidate_counts.update(normalized)

    repeated = {
        line for line, count in candidate_counts.items()
        if count >= max(2, len(page_texts) // 3)
    }

    cleaned_pages: list[str] = []
    for text in page_texts:
        kept = []
        for line in text.splitlines():
            normalized = _normalize_line_for_repetition(line)
            if normalized in repeated and (_looks_like_running_header_or_footer(line) or len(line.split()) <= 12):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept).strip())
    return cleaned_pages


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyPDF2 is required for PDF text extraction. Install requirements first.") from exc

    reader = PdfReader(str(pdf_path))
    raw_pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = _clean_page_text(text)
        if text:
            raw_pages.append(text)

    pages = _strip_repeated_margin_lines(raw_pages)
    return "\n\n".join(page for page in pages if page).strip()


COMMON_SECTION_TITLES = [
    "abstract",
    "introduction",
    "background",
    "methods",
    "method",
    "data",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "appendix",
    "references",
]


def _normalize_heading(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .:-")
    title = re.sub(r"\s*\.\s*", ".", title)
    return title


def _looks_numeric_or_table_like(line: str) -> bool:
    candidate = (line or "").strip()
    if not candidate:
        return False
    alpha_ratio = sum(ch.isalpha() for ch in candidate) / max(len(candidate), 1)
    digit_ratio = sum(ch.isdigit() for ch in candidate) / max(len(candidate), 1)
    if digit_ratio > 0.28 and alpha_ratio < 0.45:
        return True
    if candidate.count("|") >= 2:
        return True
    if re.search(r"\b(table|figure|fig\.)\b", candidate.lower()) and digit_ratio > 0.08:
        return True
    return False


def _looks_like_heading(line: str, next_line: str = "", prev_line: str = "") -> bool:
    candidate = _normalize_heading(line)
    lower = candidate.lower()
    if not candidate or len(candidate) > 160:
        return False
    if _looks_numeric_or_table_like(candidate):
        return False
    if re.fullmatch(r"[A-Za-z]?\d+(?:[.,]\d+)?(?:\s*[A-Za-z]?\d+(?:[.,]\d+)?)*", candidate):
        return False
    if re.fullmatch(r"[A-Za-z0-9_(),.=+\-/*\s]{1,30}", candidate) and any(ch in candidate for ch in "=()/"):
        return False
    if lower in COMMON_SECTION_TITLES:
        return True
    if re.fullmatch(r"(?:abstract|references|acknowledg?ments|appendix(?: [A-Z])?)", lower):
        return True
    if re.fullmatch(r"(?:\d+|[IVX]+)(?:\.\d+)*[ .]+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:/()\-]{2,120}", candidate):
        return True
    if candidate.isupper() and 4 <= len(candidate) <= 90 and sum(ch.isalpha() for ch in candidate) >= 4:
        return True
    if re.fullmatch(r"[A-Z][A-Z\- ]{4,90}", candidate):
        return True
    if prev_line == "" and next_line and len(candidate.split()) <= 14 and candidate[:1].isupper() and not candidate.endswith(('.', ',', ';', ':')):
        return True
    return False


def _merge_heading_with_following_line(lines: list[str], idx: int) -> tuple[str, int]:
    line = _normalize_heading(lines[idx])
    if idx + 1 >= len(lines):
        return line, idx
    nxt = _normalize_heading(lines[idx + 1])
    if not nxt:
        return line, idx
    if len(line) < 60 and len(nxt) < 80 and not _looks_numeric_or_table_like(nxt):
        if not _looks_like_heading(nxt) and not nxt.endswith(('.', ';')):
            merged = _normalize_heading(f"{line} {nxt}")
            if len(merged) <= 140:
                return merged, idx + 1
    return line, idx


def _clean_section_body(lines: list[str]) -> str:
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _repair_broken_words(text)
    return text.strip()


def split_into_sections(text: str) -> list[dict]:
    if not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    sections: list[dict] = []
    current_title = "front_matter"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_title, sections
        body = _clean_section_body(current_lines)
        if len(body.split()) >= 30:
            sections.append({
                "section_title": current_title,
                "section_path": current_title,
                "section_text": body,
            })
        current_lines = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        prev_line = lines[idx - 1] if idx > 0 else ""
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if _looks_like_heading(line, next_line=next_line, prev_line=prev_line):
            flush()
            merged_title, consumed_idx = _merge_heading_with_following_line(lines, idx)
            current_title = merged_title
            idx = consumed_idx + 1
            continue
        current_lines.append(line)
        idx += 1
    flush()

    if not sections:
        sections = [{"section_title": "full_text", "section_path": "full_text", "section_text": text.strip()}]
    return sections


def ingest_full_texts(manifest_df: pd.DataFrame, config: FullTextIngestConfig, limit: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    subset = manifest_df.head(limit) if limit else manifest_df

    for _, row in subset.iterrows():
        benchmark_paper_id = str(row["benchmark_paper_id"])
        source_url = str(row.get("url", ""))
        arxiv_id = extract_arxiv_id(source_url)
        pdf_url = arxiv_abs_to_pdf_url(source_url)
        pdf_path = config.pdf_dir / f"{sanitize_filename(arxiv_id)}.pdf"
        text_path = config.extracted_dir / f"{benchmark_paper_id}.txt"

        status = "missing_url"
        extracted_text = ""
        error_message = ""

        try:
            if pdf_url:
                status = download_pdf(pdf_url, pdf_path, config.user_agent)
                if config.request_pause_seconds:
                    time.sleep(config.request_pause_seconds)
                if config.overwrite or not text_path.exists():
                    extracted_text = extract_text_from_pdf(pdf_path)
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(extracted_text, encoding="utf-8")
                    status = "extracted"
                else:
                    extracted_text = text_path.read_text(encoding="utf-8")
                    status = "cached_text"
        except (HTTPError, URLError, TimeoutError) as exc:
            error_message = str(exc)
            status = "download_failed"
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)
            status = "extract_failed"

        sections = split_into_sections(extracted_text)
        rows.append(
            {
                "benchmark_paper_id": benchmark_paper_id,
                "paper_id": row.get("paper_id"),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "topic_cluster": row.get("topic_cluster", "unassigned"),
                "url": source_url,
                "pdf_url": pdf_url,
                "pdf_path": str(pdf_path) if pdf_path.exists() else "",
                "text_path": str(text_path) if text_path.exists() else "",
                "status": status,
                "error": error_message,
                "char_count": len(extracted_text),
                "section_count": len(sections),
                "sections": sections,
            }
        )

    return pd.DataFrame(rows)
