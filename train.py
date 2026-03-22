from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LABEL_COLUMN, MODELS_DIR
from models import PaperClassifier


MODEL_PATH = MODELS_DIR / "paper_classifier.joblib"


def train_classifier(train_df: pd.DataFrame, train_embeddings: np.ndarray) -> PaperClassifier:
    model = PaperClassifier()
    model.fit(train_embeddings, train_df[LABEL_COLUMN].tolist())
    return model


def save_classifier(model: PaperClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: Path = MODEL_PATH) -> PaperClassifier:
    return joblib.load(path)
