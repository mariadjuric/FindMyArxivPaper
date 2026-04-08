from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PaperFirstRerankConfig:
    paper_top_k: int = 5
    candidate_chunk_k: int = 40
    final_top_k: int = 10
    dense_weight: float = 0.7
    lexical_weight: float = 0.3


class PaperFirstDenseRerankRetriever:
    def __init__(self, config: PaperFirstRerankConfig | None = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.config = config or PaperFirstRerankConfig()
        self.model = SentenceTransformer(model_name)
        self.paper_embeddings: np.ndarray | None = None
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_texts: list[str] = []
        self.paper_ids: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_paper_ids: list[str] = []
        self.chunk_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.chunk_matrix = None

    def fit(self, chunks_df: pd.DataFrame) -> None:
        papers = chunks_df.groupby("paper_id").agg({"title": "first", "chunk_text": lambda s: " ".join(s.astype(str).tolist())}).reset_index()
        paper_texts = (papers["title"].fillna("") + " " + papers["chunk_text"].fillna("")).tolist()
        self.paper_embeddings = np.asarray(self.model.encode(paper_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.paper_ids = papers["paper_id"].astype(str).tolist()

        title = chunks_df.get("title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        section = chunks_df.get("section_title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        chunk = chunks_df["chunk_text"].fillna("").astype(str)
        self.chunk_texts = (title + " [SEP] " + section + " [SEP] " + chunk).tolist()
        self.chunk_embeddings = np.asarray(self.model.encode(self.chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
        self.chunk_paper_ids = chunks_df["paper_id"].astype(str).tolist()
        self.chunk_matrix = self.chunk_vectorizer.fit_transform(self.chunk_texts)

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.paper_embeddings is None or self.chunk_embeddings is None or self.chunk_matrix is None:
            raise RuntimeError("Retriever has not been fit.")

        query_embedding = np.asarray(self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))[0]
        paper_scores = self.paper_embeddings @ query_embedding
        paper_order = np.argsort(-paper_scores)[: self.config.paper_top_k]
        allowed_papers = {self.paper_ids[i] for i in paper_order}

        dense_scores = self.chunk_embeddings @ query_embedding
        q_lex = self.chunk_vectorizer.transform([query])
        lexical_scores = cosine_similarity(q_lex, self.chunk_matrix).ravel()

        candidates = []
        for cid, pid, dense_score, lexical_score in zip(self.chunk_ids, self.chunk_paper_ids, dense_scores, lexical_scores):
            if pid not in allowed_papers:
                continue
            combined = self.config.dense_weight * float(dense_score) + self.config.lexical_weight * float(lexical_score)
            candidates.append((cid, combined, float(dense_score), float(lexical_score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: self.config.candidate_chunk_k]
        reranked = sorted(candidates, key=lambda x: (x[1], x[2], x[3]), reverse=True)[:top_k]
        return [c[0] for c in reranked], [c[1] for c in reranked]
