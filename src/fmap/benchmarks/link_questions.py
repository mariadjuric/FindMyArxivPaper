from __future__ import annotations

from collections import defaultdict

import pandas as pd


TOPIC_TO_CLUSTER = {
    "galactic_dynamics": "galactic_dynamics",
    "distribution_functions": "distribution_functions",
    "action_angle_methods": "action_angle_methods",
    "scientific_retrieval": None,
    "scientific_qa": None,
}


def attach_gold_targets(questions_df: pd.DataFrame, chunks_df: pd.DataFrame, papers_df: pd.DataFrame) -> pd.DataFrame:
    cluster_to_papers: dict[str, list[str]] = defaultdict(list)
    cluster_to_chunks: dict[str, list[str]] = defaultdict(list)

    for _, row in papers_df.iterrows():
        cluster_to_papers[str(row.get("topic_cluster", "unassigned"))].append(str(row["benchmark_paper_id"]))
    for _, row in chunks_df.iterrows():
        cluster_to_chunks[str(row.get("topic_cluster", "unassigned"))].append(str(row["chunk_id"]))

    enriched = questions_df.copy()
    gold_papers = []
    gold_chunks = []
    for _, row in enriched.iterrows():
        cluster = TOPIC_TO_CLUSTER.get(str(row.get("topic")))
        if cluster is None:
            fallback_chunks = chunks_df.head(3)["chunk_id"].tolist()
            fallback_papers = papers_df.head(2)["benchmark_paper_id"].tolist()
            gold_papers.append(fallback_papers)
            gold_chunks.append(fallback_chunks)
            continue
        gold_papers.append(cluster_to_papers.get(cluster, [])[:3])
        gold_chunks.append(cluster_to_chunks.get(cluster, [])[:5])

    enriched["gold_papers"] = gold_papers
    enriched["gold_chunks"] = gold_chunks
    return enriched
