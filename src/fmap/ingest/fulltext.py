from __future__ import annotations

import re
import time
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


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyPDF2 is required for PDF text extraction. Install requirements first.") from exc

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def split_into_sections(text: str) -> list[dict]:
    if not text.strip():
        return []

    heading_pattern = re.compile(
        r"(?:^|\n)(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9 ,:/()\-]{2,80})\n",
        flags=re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [{"section_title": "full_text", "section_path": "full_text", "section_text": text.strip()}]

    sections: list[dict] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end].strip()
        if len(body.split()) < 20:
            continue
        sections.append({"section_title": title, "section_path": title, "section_text": body})

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
