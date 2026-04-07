from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = ranked_ids[:k]
    return float(any(item in gold_ids for item in top))


def mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_benchmark_retrieval(benchmark_df: pd.DataFrame, retriever, top_ks: Iterable[int] = (1, 3, 5, 10)) -> dict:
    top_ks = sorted(set(int(k) for k in top_ks))
    rows = []

    for _, row in benchmark_df.iterrows():
        gold_chunks = set(row.get("gold_chunks", []) or [])
        ranking, scores = retriever.search(str(row["question"]), top_k=max(top_ks))
        result = {
            "id": row["id"],
            "question": row["question"],
            "gold_chunk_count": len(gold_chunks),
        }
        for k in top_ks:
            result[f"recall@{k}"] = recall_at_k(ranking, gold_chunks, k)
            result[f"mrr@{k}"] = mrr_at_k(ranking, gold_chunks, k)
            result[f"ndcg@{k}"] = ndcg_at_k(ranking, gold_chunks, k)
            result[f"top_{k}"] = ranking[:k]
        rows.append(result)

    per_query = pd.DataFrame(rows)
    aggregate = {
        metric: float(per_query[metric].mean())
        for metric in per_query.columns
        if metric.startswith(("recall@", "mrr@", "ndcg@"))
    }
    aggregate["num_questions"] = int(len(per_query))
    return {"aggregate": aggregate, "per_query": per_query}
