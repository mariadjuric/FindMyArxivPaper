from __future__ import annotations

import re

KEEP_SECTION_KEYWORDS = {
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
    "experiment",
    "experiments",
    "observations",
    "model",
    "models",
}

DROP_SECTION_KEYWORDS = {
    "front_matter",
    "references",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "author contributions",
    "data availability",
    "code availability",
}


def normalize_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def is_equation_like_heading(title: str) -> bool:
    title = (title or "").strip()
    if not title:
        return True
    if len(title) <= 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9_().=+\-/*\s]+", title) and any(ch in title for ch in "=()/"):
        return True
    if re.fullmatch(r"[A-Za-z]\d+(?:,\s*\([A-Za-z0-9]+\))?", title):
        return True
    if re.fullmatch(r"\(?[A-Za-z0-9.\-]+\)?", title):
        return True
    if re.fullmatch(r"[A-Z0-9 ,.+\-]{6,80}", title) and sum(ch.isdigit() for ch in title) >= 3:
        return True
    return False


def is_probably_bad_section(section_title: str, section_text: str) -> bool:
    title = normalize_section_title(section_title)
    text = (section_text or "").strip()
    text_lower = text.lower()
    words = text.split()
    if title in DROP_SECTION_KEYWORDS:
        return True
    if is_equation_like_heading(section_title):
        return True
    if len(words) < 40:
        return True
    if text_lower.count("arxiv:") > 1:
        return True
    if text_lower.count("references") > 2:
        return True
    if re.search(r"\bet al\.\b", text_lower) and text_lower.count("; ") > 8:
        return True
    if re.search(r"\bdoi\b", text_lower) and text_lower.count("http") > 2:
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.22 and alpha_ratio < 0.45:
        return True
    if sum(1 for w in words if any(ch.isdigit() for ch in w)) / max(len(words), 1) > 0.35:
        return True
    if text.count("|") >= 3:
        return True
    return False


def is_preferred_section(section_title: str) -> bool:
    title = normalize_section_title(section_title)
    return any(keyword in title for keyword in KEEP_SECTION_KEYWORDS)


def chunk_quality_score(section_title: str, chunk_text: str) -> float:
    text = (chunk_text or "").strip()
    score = 0.0
    if is_preferred_section(section_title):
        score += 2.0
    if 80 <= len(text.split()) <= 260:
        score += 1.0
    if text.count("=") < 4:
        score += 0.5
    if text.lower().count("references") == 0:
        score += 0.5
    if re.search(r"\b(result|method|data|model|galactic|milky way|phase spiral|distribution function)\b", text.lower()):
        score += 1.0
    return score
