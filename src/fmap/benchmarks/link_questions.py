from __future__ import annotations

from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOPIC_TO_CLUSTER = {
    "galactic_dynamics": "galactic_dynamics",
    "distribution_functions": "distribution_functions",
    "action_angle_methods": "action_angle_methods",
    "scientific_retrieval": None,
    "scientific_qa": None,
}


def _text_blob(row: pd.Series) -> str:
    parts = [
        str(row.get("chunk_text", "")),
        str(row.get("section_title", "")),
        str(row.get("title", "")),
    ]
    return " ".join(p for p in parts if p).lower()


def _choose_chunks_for_question(question_row: pd.Series, candidate_chunks: pd.DataFrame, limit: int = 5) -> list[str]:
    if candidate_chunks.empty:
        return []

    keywords = [str(k).lower() for k in (question_row.get("keywords", []) or []) if str(k).strip()]
    question_text = str(question_row.get("question", "")).lower()
    query_terms = set(question_text.replace("?", " ").replace("-", " ").split())
    query_terms = {t for t in query_terms if len(t) > 3}

    texts = candidate_chunks.apply(_text_blob, axis=1).tolist()
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform(texts)
    query_vec = tfidf.transform([question_text])
    sim = cosine_similarity(query_vec, matrix).ravel()

    scored: list[tuple[float, str]] = []
    for idx, (_, row) in enumerate(candidate_chunks.iterrows()):
        blob = texts[idx]
        keyword_hits = sum(1 for kw in keywords if kw in blob)
        term_hits = sum(1 for term in query_terms if term in blob)
        preferred_bonus = 2.0 if bool(row.get("preferred_section", False)) else 0.0
        quality_bonus = float(row.get("chunk_quality_score", 0.0)) * 0.35
        lexical_bonus = float(sim[idx]) * 5.0
        score = lexical_bonus + (3.0 * keyword_hits) + (0.35 * term_hits) + preferred_bonus + quality_bonus
        scored.append((score, str(row.get("chunk_id"))))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [chunk_id for score, chunk_id in scored if score > 0][:limit]
    if len(chosen) < limit:
        fallbacks = candidate_chunks.sort_values(
            by=[c for c in ["preferred_section", "chunk_quality_score", "word_count"] if c in candidate_chunks.columns],
            ascending=[False, False, False][: len([c for c in ["preferred_section", "chunk_quality_score", "word_count"] if c in candidate_chunks.columns])],
        )["chunk_id"].astype(str).tolist()
        for chunk_id in fallbacks:
            if chunk_id not in chosen:
                chosen.append(chunk_id)
            if len(chosen) >= limit:
                break
    return chosen[:limit]


def attach_gold_targets(questions_df: pd.DataFrame, chunks_df: pd.DataFrame, papers_df: pd.DataFrame) -> pd.DataFrame:
    cluster_to_papers: dict[str, list[str]] = defaultdict(list)

    for _, row in papers_df.iterrows():
        cluster_to_papers[str(row.get("topic_cluster", "unassigned"))].append(str(row["benchmark_paper_id"]))

    enriched = questions_df.copy()
    gold_papers = []
    gold_chunks = []
    for _, row in enriched.iterrows():
        cluster = TOPIC_TO_CLUSTER.get(str(row.get("topic")))
        if cluster is None:
            fallback_chunks = _choose_chunks_for_question(row, chunks_df.head(12), limit=5)
            fallback_papers = papers_df.head(3)["benchmark_paper_id"].astype(str).tolist()
            gold_papers.append(fallback_papers)
            gold_chunks.append(fallback_chunks)
            continue

        candidate_papers = cluster_to_papers.get(cluster, [])[:3]
        candidate_chunks = chunks_df[chunks_df["topic_cluster"].astype(str) == cluster].copy()
        chosen_chunks = _choose_chunks_for_question(row, candidate_chunks, limit=5)
        gold_papers.append(candidate_papers)
        gold_chunks.append(chosen_chunks)

    enriched["gold_papers"] = gold_papers
    enriched["gold_chunks"] = gold_chunks
    return enriched
