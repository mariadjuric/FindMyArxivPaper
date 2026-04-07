from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    query_id: str
    ranking: list[str]
    scores: list[float]


class BM25LikeRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = None
        self.chunk_ids: list[str] = []

    def fit(self, chunks_df: pd.DataFrame) -> None:
        texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()
        self.matrix = self.vectorizer.fit_transform(texts)
        self.chunk_ids = chunks_df["chunk_id"].tolist()

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.matrix is None:
            raise RuntimeError("Retriever has not been fit.")
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        return [self.chunk_ids[i] for i in order], [float(scores[i]) for i in order]


class DenseChunkRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.chunk_ids: list[str] = []
        self.model_name = model_name

    def fit(self, chunks_df: pd.DataFrame) -> None:
        texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()
        self.embeddings = np.asarray(self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        self.chunk_ids = chunks_df["chunk_id"].tolist()

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        if self.embeddings is None:
            raise RuntimeError("Retriever has not been fit.")
        query_embedding = np.asarray(self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))[0]
        scores = self.embeddings @ query_embedding
        order = np.argsort(-scores)[:top_k]
        return [self.chunk_ids[i] for i in order], [float(scores[i]) for i in order]
