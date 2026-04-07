from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

TOPIC_CLUSTER_KEYWORDS = {
    "galactic_dynamics": [
        "galactic",
        "milky way",
        "disc",
        "disk",
        "spiral",
        "phase-space",
        "phase space",
        "bending",
        "satellite",
        "disequilibrium",
        "phase mixing",
        "vertical",
    ],
    "distribution_functions": [
        "distribution function",
        "distribution functions",
        "fokker",
        "diffusion",
        "inference",
        "selection function",
        "selection effects",
        "equilibrium",
        "non-equilibrium",
        "phase-space density",
        "dynamical model",
        "survey bias",
    ],
    "action_angle_methods": [
        "action-angle",
        "action angle",
        "hamiltonian",
        "symplectic",
        "orbit",
        "orbital",
        "canonical",
        "frequency",
        "frequencies",
        "integrable",
    ],
}

NEGATIVE_MATCH_TERMS = [
    "april fools",
    "cannabinoid",
    "cannabis",
    "galactic constellations",
    "wonderful astronomer",
    "phermon",
    "crocs data release",
    "star wars",
    "cat-driven",
    "purr-fect",
    "balanced hybrid",
    "space nuggets",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def stable_paper_id(title: str, url: str = "", prefix: str = "astrobench") -> str:
    seed = f"{_normalize_text(title).lower()}::{(url or '').strip().lower()}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def infer_topic_cluster(title: str, abstract: str) -> str:
    haystack = f"{title} {abstract}".lower()
    if any(term in haystack for term in NEGATIVE_MATCH_TERMS):
        return "unassigned"
    scores: dict[str, int] = {}
    for cluster, keywords in TOPIC_CLUSTER_KEYWORDS.items():
        scores[cluster] = sum(1 for kw in keywords if kw in haystack)
    best_cluster = max(scores, key=scores.get)
    return best_cluster if scores[best_cluster] > 0 else "unassigned"


def build_corpus_manifest(
    df: pd.DataFrame,
    limit: int = 40,
    include_clusters: Iterable[str] | None = None,
    seed_path: Path | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work["topic_cluster"] = [infer_topic_cluster(t, a) for t, a in zip(work["title"], work["abstract"])]
    work["benchmark_paper_id"] = [stable_paper_id(t, u) for t, u in zip(work["title"], work.get("url", ""))]
    work["included_in_benchmark"] = False

    if seed_path is not None and seed_path.exists():
        seeds = pd.read_csv(seed_path)
        if include_clusters is not None:
            include_clusters = set(include_clusters)
            seeds = seeds[seeds["cluster"].isin(include_clusters)]
        selected = work[work["paper_id"].isin(seeds["paper_id"].tolist())].copy()
        seed_cluster_map = dict(zip(seeds["paper_id"], seeds["cluster"]))
        selected["topic_cluster"] = selected["paper_id"].map(seed_cluster_map).fillna(selected["topic_cluster"])
        selected["included_in_benchmark"] = True
        selected = selected.merge(
            seeds[["paper_id", "priority", "notes"]],
            on="paper_id",
            how="left",
        )
        selected = selected.sort_values(["topic_cluster", "priority", "year", "published"], ascending=[True, True, False, False])
        selected = selected.drop_duplicates(subset=["paper_id"], keep="first")
        if limit:
            selected = selected.head(limit)
    else:
        work = work[work["topic_cluster"] != "unassigned"].copy()
        work["included_in_benchmark"] = True

        if include_clusters is not None:
            include_clusters = set(include_clusters)
            work = work[work["topic_cluster"].isin(include_clusters)]

        selected_parts: list[pd.DataFrame] = []
        selected_clusters = [
            "galactic_dynamics",
            "distribution_functions",
            "action_angle_methods",
        ]
        per_cluster = max(limit // max(len(selected_clusters), 1), 1)

        for cluster in selected_clusters:
            part = work[work["topic_cluster"] == cluster].copy()
            if part.empty:
                continue
            sort_cols = [c for c in ["year", "published"] if c in part.columns]
            ascending = [False] * len(sort_cols)
            if sort_cols:
                part = part.sort_values(sort_cols, ascending=ascending)
            selected_parts.append(part.head(per_cluster))

        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else work.head(0).copy()
        if len(selected) < limit:
            used_ids = set(selected.get("benchmark_paper_id", pd.Series(dtype=str)).tolist())
            remainder = work[~work["benchmark_paper_id"].isin(used_ids)].copy()
            sort_cols = [c for c in ["year", "published"] if c in remainder.columns]
            if sort_cols:
                remainder = remainder.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            selected = pd.concat([selected, remainder.head(limit - len(selected))], ignore_index=True)

    keep_cols = [
        "benchmark_paper_id",
        "paper_id",
        "title",
        "authors",
        "abstract",
        "category",
        "primary_category",
        "published",
        "updated",
        "url",
        "year",
        "topic_cluster",
        "included_in_benchmark",
        "priority",
        "notes",
    ]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    return selected[keep_cols].reset_index(drop=True)


def save_corpus_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
