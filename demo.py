from __future__ import annotations

import argparse

from config import EMBEDDER_NAME
from data import load_dataset
from models import PaperEmbedder
from search import semantic_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic search over scientific papers.")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset()
    embedder = PaperEmbedder(EMBEDDER_NAME)
    embeddings = embedder.encode(df["abstract"].tolist())
    results = semantic_search(args.query, df, embeddings, embedder, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['title']} | {item['category']} | score={item['score']:.3f}")
        print(f"   {item['abstract']}\n")


if __name__ == "__main__":
    main()
