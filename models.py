from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


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
    classifier: Pipeline | None = None
    label_encoder: LabelEncoder | None = None
    vectorizer_max_features: int = 8000
    vectorizer_ngram_range: tuple[int, int] = (1, 2)
    classifier_c: float = 1.0
    text_field_name: str = field(default="title + abstract")

    def fit(self, texts: list[str], y: list[str]) -> None:
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classifier = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.vectorizer_ngram_range,
                        max_features=self.vectorizer_max_features,
                        strip_accents="unicode",
                        lowercase=True,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LinearSVC(C=self.classifier_c)),
            ]
        )
        self.classifier.fit(texts, y_encoded)

    def predict(self, texts: list[str]) -> list[str]:
        if self.classifier is None or self.label_encoder is None:
            raise RuntimeError("Classifier has not been fit.")
        preds = self.classifier.predict(texts)
        return self.label_encoder.inverse_transform(preds).tolist()
