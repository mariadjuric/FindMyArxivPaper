from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FULLTEXT_CSV = ROOT / "data" / "processed" / "paper_fulltext.csv"
SECTIONS_JSON = ROOT / "data" / "processed" / "paper_sections.json"
CHUNKS_CSV = ROOT / "data" / "processed" / "paper_section_chunks.csv"
OUT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
HTML_PATH = OUT_DIR / "fulltext_report.html"


PREFERRED_TITLES = {
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
    "summary",
    "analysis",
    "observations",
}


def _norm_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def _is_preferred_title(title: str) -> bool:
    t = _norm_title(title)
    return any(key in t for key in PREFERRED_TITLES)


def make_figures(fulltext: pd.DataFrame, chunks: pd.DataFrame) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    status_counts = fulltext["status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    status_counts.plot(kind="bar", ax=ax, color="#627f2f")
    ax.set_title("Full-text ingestion status")
    ax.set_ylabel("Paper count")
    fig.tight_layout()
    p = FIG_DIR / "fulltext_status_counts.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fulltext["char_count"].plot(kind="hist", bins=12, ax=ax, color="#5ab3ff")
    ax.set_title("Extracted full-text character counts")
    ax.set_xlabel("Characters")
    fig.tight_layout()
    p = FIG_DIR / "fulltext_char_count_hist.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fulltext["section_count"].plot(kind="hist", bins=10, ax=ax, color="#ff8a3d")
    ax.set_title("Detected section counts per paper")
    ax.set_xlabel("Section count")
    fig.tight_layout()
    p = FIG_DIR / "fulltext_section_count_hist.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    chunk_counts = chunks.groupby("paper_id")["chunk_id"].count().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    chunk_counts.plot(kind="bar", ax=ax, color="#b57cff")
    ax.set_title("Section-aware chunk count per paper")
    ax.set_ylabel("Chunks")
    ax.set_xlabel("benchmark paper id")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    p = FIG_DIR / "section_chunk_count_per_paper.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    if "preferred_section" in chunks.columns:
        pref_counts = chunks["preferred_section"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5.8, 4))
        pref_counts.plot(kind="bar", ax=ax, color=["#94a3b8", "#22c55e"])
        ax.set_title("Chunk keep mix: semantic vs other sections")
        ax.set_xlabel("preferred section")
        ax.set_ylabel("Chunk count")
        fig.tight_layout()
        p = FIG_DIR / "preferred_section_chunk_mix.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    return paths


def _best_sections(sections: list[dict], limit: int = 6) -> list[dict]:
    preferred = [s for s in sections if _is_preferred_title(str(s.get("section_title", "")))]
    other = [s for s in sections if s not in preferred]
    chosen = preferred[:limit]
    if len(chosen) < limit:
        chosen.extend(other[: limit - len(chosen)])
    return chosen[:limit]


def _paper_chunk_examples(chunks: pd.DataFrame, paper_id: str, limit: int = 4) -> list[str]:
    if chunks.empty:
        return []
    subset = chunks[chunks["paper_id"].astype(str) == str(paper_id)].copy()
    if subset.empty:
        return []
    sort_cols = [c for c in ["chunk_quality_score", "preferred_section", "word_count"] if c in subset.columns]
    if sort_cols:
        subset = subset.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    rows = []
    for _, rec in subset.head(limit).iterrows():
        preview = html.escape(str(rec.get("chunk_text", ""))[:420])
        section_title = html.escape(str(rec.get("section_title", "")))
        meta = []
        if "word_count" in rec:
            meta.append(f"{int(rec['word_count'])} words")
        if "chunk_quality_score" in rec:
            meta.append(f"score {float(rec['chunk_quality_score']):.1f}")
        rows.append(f"<li><strong>{section_title}</strong> <em>({' | '.join(meta)})</em><br><pre>{preview}</pre></li>")
    return rows


def main() -> None:
    fulltext = pd.read_csv(FULLTEXT_CSV)
    chunks = pd.read_csv(CHUNKS_CSV)
    with open(SECTIONS_JSON, "r", encoding="utf-8") as f:
        section_rows = json.load(f)

    figure_paths = make_figures(fulltext, chunks)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cards = []
    for row in section_rows[:12]:
        title = html.escape(str(row.get("title", "")))
        status = html.escape(str(row.get("status", "")))
        sections = row.get("sections", []) or []
        chosen_sections = _best_sections(sections, limit=6)
        section_list = []
        for sec in chosen_sections:
            sec_title = html.escape(str(sec.get("section_title", "")))
            sec_preview = html.escape(str(sec.get("section_text", ""))[:500])
            section_list.append(f"<li><strong>{sec_title}</strong><br><pre>{sec_preview}</pre></li>")

        chunk_examples = _paper_chunk_examples(chunks, str(row.get("benchmark_paper_id", "")), limit=4)
        cards.append(
            f"<section class='card'>"
            f"<h2>{title}</h2>"
            f"<p><strong>Status:</strong> {status} | <strong>Sections:</strong> {len(sections)} | <strong>Chars:</strong> {row.get('char_count', 0)}</p>"
            f"<p><strong>PDF:</strong> {html.escape(str(row.get('pdf_path', '')))}</p>"
            f"<h3>Best extracted sections</h3>"
            f"<ul>{''.join(section_list)}</ul>"
            f"<h3>Best retrieval chunks</h3>"
            f"<ul>{''.join(chunk_examples) if chunk_examples else '<li>No chunks kept.</li>'}</ul>"
            f"</section>"
        )

    imgs = ''.join([f"<div class='figure'><img src='../figures/{p.name}' alt='{p.name}'></div>" for p in figure_paths])
    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>FMAP Full-text Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; line-height: 1.45; color: #111827; }}
    .figure img {{ max-width: 840px; border: 1px solid #ddd; margin-bottom: 20px; border-radius: 8px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    pre {{ white-space: pre-wrap; background: #fafafa; padding: 10px; border-radius: 6px; }}
    h1, h2, h3 {{ color: #0f172a; }}
    ul {{ padding-left: 20px; }}
  </style>
</head>
<body>
  <h1>FMAP full-text ingestion report</h1>
  <p>This report shows ingestion status, extraction quality indicators, detected sections, and chunking diagnostics for the curated benchmark corpus.</p>
  {imgs}
  <h1>Sample extracted papers</h1>
  {''.join(cards)}
</body>
</html>
"""
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote report to {HTML_PATH}")


if __name__ == "__main__":
    main()
