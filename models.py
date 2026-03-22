from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class PaperEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


@dataclass
class PaperClassifier:
    classifier: LogisticRegression | None = None
    label_encoder: LabelEncoder | None = None

    def fit(self, X: np.ndarray, y: list[str]) -> None:
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classifier = LogisticRegression(max_iter=2000)
        self.classifier.fit(X, y_encoded)

    def predict(self, X: np.ndarray) -> list[str]:
        if self.classifier is None or self.label_encoder is None:
            raise RuntimeError("Classifier has not been fit.")
        preds = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Classifier has not been fit.")
        return self.classifier.predict_proba(X)
