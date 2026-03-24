from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    model_name: str = field(default="tfidf + linear_svc")

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


@dataclass
class DeepPaperClassifier:
    transformer_model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    random_state: int = 42
    text_field_name: str = field(default="title + abstract")
    model_name: str = field(default="transformer_finetune")
    tokenizer = None
    model = None
    label_encoder: LabelEncoder | None = None
    device: str | None = None

    def initialize(self, labels: list[str]) -> None:
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.transformer_model_name,
            num_labels=len(self.label_encoder.classes_),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, texts: list[str]) -> list[str]:
        if self.model is None or self.tokenizer is None or self.label_encoder is None or self.device is None:
            raise RuntimeError("Deep classifier has not been initialized/trained.")
        self.model.eval()
        predictions: list[int] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        return self.label_encoder.inverse_transform(predictions).tolist()

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            raise RuntimeError("Deep classifier has not been initialized/trained.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        np.save(output_dir / "label_classes.npy", self.label_encoder.classes_)
        metadata = {
            "transformer_model_name": self.transformer_model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "text_field_name": self.text_field_name,
            "model_name": self.model_name,
        }
        import json
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, output_dir: Path) -> "DeepPaperClassifier":
        import json
        with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model = cls(**{k: metadata[k] for k in [
            "transformer_model_name", "max_length", "batch_size", "learning_rate", "weight_decay", "epochs", "text_field_name", "model_name"
        ] if k in metadata})
        model.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        model.label_encoder = LabelEncoder()
        model.label_encoder.classes_ = np.load(output_dir / "label_classes.npy", allow_pickle=True)
        model.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.model.to(model.device)
        return model
