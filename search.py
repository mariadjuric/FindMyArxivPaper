from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_COLUMN, MODEL_TEXT_COLUMN, TEXT_COLUMN, TITLE_COLUMN


def semantic_search(query: str, df: pd.DataFrame, embeddings: np.ndarray, embedder, top_k: int = 5) -> list[dict]:
    query_embedding = embedder.encode([query])[0]
    scores = embeddings @ query_embedding
    best_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in best_idx:
        row = df.iloc[idx]
        results.append(
            {
                "title": row[TITLE_COLUMN],
                "category": row[LABEL_COLUMN],
                "abstract": row[TEXT_COLUMN],
                "combined_text": row[MODEL_TEXT_COLUMN],
                "score": float(scores[idx]),
            }
        )
    return results
