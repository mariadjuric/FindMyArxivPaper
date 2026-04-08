from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PaperFirstConfig:
    paper_top_k: int = 5
    final_top_k: int = 10


class PaperFirstBM25Retriever:
    def __init__(self, config: PaperFirstConfig | None = None) -> None:
        self.config = config or PaperFirstConfig()
        self.paper_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.chunk_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.paper_matrix = None
        self.chunk_matrix = None
        self.paper_ids: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_paper_ids: list[str] = []

    def fit(self, chunks_df: pd.DataFrame) -> None:
        papers = chunks_df.groupby("paper_id").agg({"title": "first", "chunk_text": lambda s: " ".join(s.astype(str).tolist())}).reset_index()
        paper_texts = (papers["title"].fillna("") + " " + papers["chunk_text"].fillna("")).tolist()
        self.paper_matrix = self.paper_vectorizer.fit_transform(paper_texts)
        self.paper_ids = papers["paper_id"].tolist()

        title = chunks_df.get("title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        section = chunks_df.get("section_title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        chunk = chunks_df["chunk_text"].fillna("").astype(str)
        chunk_texts = (title + " " + section + " " + chunk).tolist()
        self.chunk_matrix = self.chunk_vectorizer.fit_transform(chunk_texts)
        self.chunk_ids = chunks_df["chunk_id"].tolist()
        self.chunk_paper_ids = chunks_df["paper_id"].astype(str).tolist()

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.paper_matrix is None or self.chunk_matrix is None:
            raise RuntimeError("Retriever has not been fit.")
        q_paper = self.paper_vectorizer.transform([query])
        paper_scores = cosine_similarity(q_paper, self.paper_matrix).ravel()
        paper_order = np.argsort(-paper_scores)[: self.config.paper_top_k]
        allowed_papers = {self.paper_ids[i] for i in paper_order}

        q_chunk = self.chunk_vectorizer.transform([query])
        chunk_scores = cosine_similarity(q_chunk, self.chunk_matrix).ravel()
        masked = [(cid, pid, float(score)) for cid, pid, score in zip(self.chunk_ids, self.chunk_paper_ids, chunk_scores) if pid in allowed_papers]
        masked.sort(key=lambda x: x[2], reverse=True)
        masked = masked[:top_k]
        return [m[0] for m in masked], [m[2] for m in masked]


class PaperFirstDenseRetriever:
    def __init__(self, config: PaperFirstConfig | None = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.config = config or PaperFirstConfig()
        self.model = SentenceTransformer(model_name)
        self.paper_embeddings: np.ndarray | None = None
        self.chunk_embeddings: np.ndarray | None = None
        self.paper_ids: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_paper_ids: list[str] = []

    def fit(self, chunks_df: pd.DataFrame) -> None:
        papers = chunks_df.groupby("paper_id").agg({"title": "first", "chunk_text": lambda s: " ".join(s.astype(str).tolist())}).reset_index()
        paper_texts = (papers["title"].fillna("") + " " + papers["chunk_text"].fillna("")).tolist()
        self.paper_embeddings = np.asarray(self.model.encode(paper_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.paper_ids = papers["paper_id"].tolist()

        title = chunks_df.get("title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        section = chunks_df.get("section_title", pd.Series([""] * len(chunks_df))).fillna("").astype(str)
        chunk = chunks_df["chunk_text"].fillna("").astype(str)
        chunk_texts = (title + " [SEP] " + section + " [SEP] " + chunk).tolist()
        self.chunk_embeddings = np.asarray(self.model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.chunk_ids = chunks_df["chunk_id"].tolist()
        self.chunk_paper_ids = chunks_df["paper_id"].astype(str).tolist()

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.paper_embeddings is None or self.chunk_embeddings is None:
            raise RuntimeError("Retriever has not been fit.")
        query_embedding = np.asarray(self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))[0]
        paper_scores = self.paper_embeddings @ query_embedding
        paper_order = np.argsort(-paper_scores)[: self.config.paper_top_k]
        allowed_papers = {self.paper_ids[i] for i in paper_order}

        chunk_scores = self.chunk_embeddings @ query_embedding
        masked = [(cid, pid, float(score)) for cid, pid, score in zip(self.chunk_ids, self.chunk_paper_ids, chunk_scores) if pid in allowed_papers]
        masked.sort(key=lambda x: x[2], reverse=True)
        masked = masked[:top_k]
        return [m[0] for m in masked], [m[2] for m in masked]
