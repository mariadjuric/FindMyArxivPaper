from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_PATH, EMBEDDER_NAME, PERFECT_DATA_PATH
from data import load_dataset
from models import PaperEmbedder
from search import semantic_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic search over scientific papers.")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--input", type=str, default=None, help="Optional path to a CSV dataset")
    parser.add_argument("--use-perfect", action="store_true", help="Use the perfect synthetic dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.input).expanduser().resolve() if args.input else (PERFECT_DATA_PATH if args.use-perfect else DATA_PATH)
    df = load_dataset(dataset_path)
    embedder = PaperEmbedder(EMBEDDER_NAME)
    embeddings = embedder.encode(df["combined_text"].tolist())
    results = semantic_search(args.query, df, embeddings, embedder, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['title']} | {item['category']} | score={item['score']:.3f}")
        print(f"   {item['abstract']}\n")


if __name__ == "__main__":
    main()
