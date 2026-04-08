from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PREFERRED_SECTION_HINTS = (
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "summary",
    "analysis",
    "observations",
    "data",
)


@dataclass
class HybridFullTextConfig:
    paper_top_k: int = 6
    candidate_chunk_k: int = 80
    final_top_k: int = 10
    paper_weight: float = 0.25
    dense_weight: float = 0.30
    lexical_weight: float = 0.20
    quality_weight: float = 0.15
    section_weight: float = 0.10


class HybridFullTextRetriever:
    def __init__(self, config: HybridFullTextConfig | None = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.config = config or HybridFullTextConfig()
        self.model = SentenceTransformer(model_name)
        self.paper_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.chunk_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.paper_matrix = None
        self.chunk_matrix = None
        self.paper_embeddings: np.ndarray | None = None
        self.chunk_embeddings: np.ndarray | None = None
        self.paper_ids: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_paper_ids: list[str] = []
        self.paper_score_lookup: dict[str, float] = {}
        self.chunk_quality: list[float] = []
        self.section_bonus: list[float] = []

    def fit(self, chunks_df: pd.DataFrame) -> None:
        papers = chunks_df.groupby("paper_id").agg({"title": "first", "chunk_text": lambda s: " ".join(s.astype(str).tolist())}).reset_index()
        paper_texts = (papers["title"].fillna("") + " " + papers["chunk_text"].fillna("")).tolist()
        self.paper_matrix = self.paper_vectorizer.fit_transform(paper_texts)
        self.paper_embeddings = np.asarray(self.model.encode(paper_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.paper_ids = papers["paper_id"].astype(str).tolist()

        title = chunks_df.get("title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        section = chunks_df.get("section_title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        chunk = chunks_df["chunk_text"].fillna("").astype(str)
        chunk_texts = (title + " [SEP] " + section + " [SEP] " + chunk).tolist()
        self.chunk_matrix = self.chunk_vectorizer.fit_transform(chunk_texts)
        self.chunk_embeddings = np.asarray(self.model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
        self.chunk_paper_ids = chunks_df["paper_id"].astype(str).tolist()
        self.chunk_quality = chunks_df.get("chunk_quality_score", pd.Series([0.0] * len(chunks_df))).fillna(0.0).astype(float).tolist()
        self.section_bonus = [
            1.0 if any(h in str(title).lower() for h in PREFERRED_SECTION_HINTS) else 0.0
            for title in chunks_df.get("section_title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        ]

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.paper_matrix is None or self.chunk_matrix is None or self.paper_embeddings is None or self.chunk_embeddings is None:
            raise RuntimeError("Retriever has not been fit.")

        q_lex_paper = self.paper_vectorizer.transform([query])
        paper_lex = cosine_similarity(q_lex_paper, self.paper_matrix).ravel()
        q_dense = np.asarray(self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))[0]
        paper_dense = self.paper_embeddings @ q_dense
        combined_paper = 0.5 * paper_lex + 0.5 * paper_dense
        paper_order = np.argsort(-combined_paper)[: self.config.paper_top_k]
        allowed_papers = {self.paper_ids[i] for i in paper_order}
        self.paper_score_lookup = {self.paper_ids[i]: float(combined_paper[i]) for i in paper_order}

        q_lex_chunk = self.chunk_vectorizer.transform([query])
        chunk_lex = cosine_similarity(q_lex_chunk, self.chunk_matrix).ravel()
        chunk_dense = self.chunk_embeddings @ q_dense

        max_quality = max(self.chunk_quality) if self.chunk_quality else 1.0
        candidates = []
        for cid, pid, lex, dense, quality, sec_bonus in zip(
            self.chunk_ids,
            self.chunk_paper_ids,
            chunk_lex,
            chunk_dense,
            self.chunk_quality,
            self.section_bonus,
        ):
            if pid not in allowed_papers:
                continue
            paper_score = self.paper_score_lookup.get(pid, 0.0)
            quality_norm = float(quality) / max(max_quality, 1.0)
            score = (
                self.config.paper_weight * float(paper_score)
                + self.config.dense_weight * float(dense)
                + self.config.lexical_weight * float(lex)
                + self.config.quality_weight * quality_norm
                + self.config.section_weight * float(sec_bonus)
            )
            candidates.append((cid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: self.config.candidate_chunk_k]
        final = candidates[:top_k]
        return [c[0] for c in final], [c[1] for c in final]
