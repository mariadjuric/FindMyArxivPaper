from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.fmap.evaluation.retrieval_eval import evaluate_benchmark_retrieval
from src.fmap.retrieval.baselines import BM25LikeRetriever, DenseChunkRetriever

ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNKS_PATH = ROOT / "data" / "processed" / "paper_chunks.csv"
QUESTIONS_PATH = ROOT / "benchmarks" / "astrophysics_qa" / "questions.linked.json"
OUT_DIR = ROOT / "outputs" / "metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--label", type=str, default="benchmark_retrieval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks)
    chunks = pd.read_csv(chunks_path)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = pd.DataFrame(json.load(f))

    bm25 = BM25LikeRetriever()
    bm25.fit(chunks)
    bm25_results = evaluate_benchmark_retrieval(questions, bm25, top_ks=(1, 3, 5, 10))

    dense = DenseChunkRetriever()
    dense.fit(chunks)
    dense_results = evaluate_benchmark_retrieval(questions, dense, top_ks=(1, 3, 5, 10))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{args.label}_bm25.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": bm25_results["aggregate"], "per_query": bm25_results["per_query"].to_dict(orient="records")}, f, indent=2)
    with open(OUT_DIR / f"{args.label}_dense.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": dense_results["aggregate"], "per_query": dense_results["per_query"].to_dict(orient="records")}, f, indent=2)

    print(f"Chunk file: {chunks_path}")
    print("BM25-like aggregate metrics:")
    print(json.dumps(bm25_results["aggregate"], indent=2))
    print("Dense aggregate metrics:")
    print(json.dumps(dense_results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
