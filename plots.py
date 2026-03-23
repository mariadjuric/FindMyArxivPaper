from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from config import FIGURES_DIR, LABEL_COLUMN, PUBLISHED_COLUMN, TITLE_COLUMN, UMAP_MIN_DIST, UMAP_N_NEIGHBORS, UMAP_RANDOM_STATE

sns.set_theme(style="whitegrid")


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, dodge=False, palette="viridis", legend=False)
    plt.title("Paper Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=200)
    plt.close()


def plot_year_distribution(df: pd.DataFrame) -> None:
    if PUBLISHED_COLUMN not in df.columns:
        return
    years = pd.to_datetime(df[PUBLISHED_COLUMN], errors="coerce").dt.year.dropna().astype(int)
    if years.empty:
        return
    counts = years.value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#7dd3fc")
    plt.title("Paper Count by Published Year")
    plt.xlabel("Published year")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "year_distribution.png", dpi=200)
    plt.close()


def plot_embedding_projection(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(df) < 2:
        return
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, max(2, len(df) - 1)),
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    projection = reducer.fit_transform(embeddings)
    plot_df = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1], LABEL_COLUMN: df[LABEL_COLUMN].values, TITLE_COLUMN: df[TITLE_COLUMN].values})

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x="x", y="y", hue=LABEL_COLUMN, s=10, palette="tab10", alpha=0.72, linewidth=0)
    plt.title(f"2D UMAP Projection of Paper Embeddings (n_neighbors={min(UMAP_N_NEIGHBORS, max(2, len(df) - 1))}, min_dist={UMAP_MIN_DIST})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "embedding_projection.png", dpi=220)
    plt.close()


def plot_confusion_matrix(labels: list[str], confusion: list[list[int]]) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
