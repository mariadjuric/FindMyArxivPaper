from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fmap.evaluation.retrieval_eval import evaluate_benchmark_retrieval
from src.fmap.retrieval.baselines import BM25LikeRetriever, DenseChunkRetriever

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "benchmarks" / "astrophysics_qa" / "questions.linked.json"
ABSTRACT_CHUNKS = ROOT / "data" / "processed" / "paper_chunks.csv"
FULLTEXT_CHUNKS = ROOT / "data" / "processed" / "paper_section_chunks.csv"
OUT_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"


def run_eval(chunks_path: Path, label: str) -> dict:
    chunks = pd.read_csv(chunks_path)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = pd.DataFrame(json.load(f))

    bm25 = BM25LikeRetriever()
    bm25.fit(chunks)
    bm25_results = evaluate_benchmark_retrieval(questions, bm25, top_ks=(1, 3, 5, 10))

    dense = DenseChunkRetriever()
    dense.fit(chunks)
    dense_results = evaluate_benchmark_retrieval(questions, dense, top_ks=(1, 3, 5, 10))

    return {
        "label": label,
        "bm25": bm25_results["aggregate"],
        "dense": dense_results["aggregate"],
    }


def make_plot(results: list[dict]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["recall@10", "mrr@10", "ndcg@10"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    series = [
        ("abstract_bm25", [results[0]["bm25"][m] for m in metrics], "#64748b"),
        ("abstract_dense", [results[0]["dense"][m] for m in metrics], "#94a3b8"),
        ("fulltext_bm25", [results[1]["bm25"][m] for m in metrics], "#627f2f"),
        ("fulltext_dense", [results[1]["dense"][m] for m in metrics], "#a3be4c"),
    ]
    for idx, (name, vals, color) in enumerate(series):
        ax.bar(x + (idx - 1.5) * width, vals, width, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Abstract vs full-text retrieval benchmark comparison")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "chunk_retrieval_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    results = [
        run_eval(ABSTRACT_CHUNKS, "abstract_chunks"),
        run_eval(FULLTEXT_CHUNKS, "fulltext_chunks"),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "chunk_retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    fig_path = make_plot(results)
    print(json.dumps(results, indent=2))
    print(f"Saved comparison plot to {fig_path}")


if __name__ == "__main__":
    main()
