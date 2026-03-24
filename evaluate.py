from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import LABEL_COLUMN, METRICS_DIR, MODEL_TEXT_COLUMN, TOP_K
from utils import save_json


def evaluate_classification(model, test_df: pd.DataFrame) -> dict:
    y_true = test_df[LABEL_COLUMN].tolist()
    y_pred = model.predict(test_df[MODEL_TEXT_COLUMN].tolist())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "labels": sorted(set(y_true) | set(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred))).tolist(),
        "model_text_column": MODEL_TEXT_COLUMN,
        "classifier": getattr(model, "model_name", model.__class__.__name__),
    }
    save_json(metrics, METRICS_DIR / "classification_metrics.json")
    return metrics


def precision_at_k(query_label: str, retrieved_labels: Sequence[str], k: int) -> float:
    top = list(retrieved_labels)[:k]
    if not top:
        return 0.0
    return sum(label == query_label for label in top) / len(top)


def evaluate_retrieval(df: pd.DataFrame, embeddings: np.ndarray, top_k: int = TOP_K) -> dict:
    labels = df[LABEL_COLUMN].tolist()
    similarities = embeddings @ embeddings.T
    precisions = []

    for i in range(len(df)):
        ranked_idx = np.argsort(-similarities[i])
        ranked_idx = [idx for idx in ranked_idx if idx != i][:top_k]
        retrieved_labels = [labels[idx] for idx in ranked_idx]
        precisions.append(precision_at_k(labels[i], retrieved_labels, top_k))

    metrics = {
        f"precision_at_{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        "num_queries": len(df),
        "text_column": MODEL_TEXT_COLUMN,
    }
    save_json(metrics, METRICS_DIR / "retrieval_metrics.json")
    return metrics
