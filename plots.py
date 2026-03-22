from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from config import FIGURES_DIR, LABEL_COLUMN, TITLE_COLUMN

sns.set_theme(style="whitegrid")


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("Paper Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=200)
    plt.close()


def plot_embedding_projection(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(df) < 2:
        return
    reducer = PCA(n_components=2)
    projection = reducer.fit_transform(embeddings)
    plot_df = pd.DataFrame({
        "x": projection[:, 0],
        "y": projection[:, 1],
        LABEL_COLUMN: df[LABEL_COLUMN].values,
        TITLE_COLUMN: df[TITLE_COLUMN].values,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue=LABEL_COLUMN, s=90, palette="tab10")
    plt.title("2D PCA Projection of Paper Embeddings")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "embedding_projection.png", dpi=200)
    plt.close()


def plot_confusion_matrix(labels: list[str], confusion: list[list[int]]) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
