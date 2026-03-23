from __future__ import annotations

import argparse
from pathlib import Path

from arxiv_data import ArxivFetchError, fetch_arxiv_dataset
from config import ARXIV_DATA_PATH, DATA_PATH, EMBEDDER_NAME, PERFECT_DATA_PATH, TOP_K, ensure_directories
from data import load_dataset, make_splits
from evaluate import evaluate_classification, evaluate_retrieval
from models import PaperEmbedder
from plots import plot_confusion_matrix, plot_embedding_projection, plot_label_distribution
from search import semantic_search
from site_builder import build_site
from train import save_classifier, train_classifier
from utils import print_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SciPaper pipeline.")
    parser.add_argument("--source", choices=["synthetic", "perfect", "arxiv", "csv"], default="synthetic")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV when --source csv is used")
    parser.add_argument("--max-results", type=int, default=500, help="Max arXiv results when --source arxiv")
    parser.add_argument("--categories", type=str, default="", help="Comma-separated arXiv categories for --source arxiv")
    parser.add_argument("--skip-site", action="store_true", help="Skip static site generation")
    return parser.parse_args()


def resolve_dataset(args: argparse.Namespace) -> Path:
    if args.source == "perfect":
        return PERFECT_DATA_PATH
    if args.source == "csv":
        if not args.input:
            raise ValueError("--input is required when --source csv")
        return Path(args.input).expanduser().resolve()
    if args.source == "arxiv":
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        print_section("Fetching arXiv data")
        try:
            df = fetch_arxiv_dataset(categories=categories or None, max_results=args.max_results, output_path=ARXIV_DATA_PATH)
        except ArxivFetchError as exc:
            raise RuntimeError(
                "arXiv fetch failed. If the API is slow or rate-limiting you, rerun later or reuse a cached "
                f"dataset at {ARXIV_DATA_PATH}. Details: {exc}"
            ) from exc
        print(f"Using arXiv dataset with {len(df)} rows at {ARXIV_DATA_PATH}")
        return ARXIV_DATA_PATH
    return DATA_PATH


def main() -> None:
    args = parse_args()
    ensure_directories()

    dataset_path = resolve_dataset(args)

    print_section("Loading data")
    df = load_dataset(dataset_path)
    print(f"Loaded {len(df)} papers from {dataset_path}")

    print_section("Creating splits")
    train_df, test_df = make_splits(df)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    print_section("Building embeddings")
    embedder = PaperEmbedder(EMBEDDER_NAME)
    all_embeddings = embedder.encode(df["combined_text"].tolist())
    print(f"Embedding shape: {all_embeddings.shape}")

    print_section("Training classifier")
    classifier = train_classifier(train_df)
    save_classifier(classifier)
    print("Saved classifier to outputs/models/")

    print_section("Evaluating classifier")
    clf_metrics = evaluate_classification(classifier, test_df)
    print(f"Accuracy: {clf_metrics['accuracy']:.3f}")
    print(f"Macro F1: {clf_metrics['macro_f1']:.3f}")

    print_section("Evaluating retrieval")
    retrieval_metrics = evaluate_retrieval(df, all_embeddings, top_k=TOP_K)
    print(f"Precision@{TOP_K}: {retrieval_metrics[f'precision_at_{TOP_K}']:.3f}")

    print_section("Generating plots")
    plot_label_distribution(df)
    plot_embedding_projection(df, all_embeddings)
    plot_confusion_matrix(clf_metrics["labels"], clf_metrics["confusion_matrix"])
    print("Saved plots to outputs/figures/")

    if not args.skip_site:
        print_section("Building static site")
        site_index = build_site(df, all_embeddings)
        print(f"Built interactive site at {site_index}")

    print_section("Example semantic search")
    query = "astrophysical survey of galaxies and cosmology" if args.source == "arxiv" else "transformer models for scientific document retrieval"
    results = semantic_search(query, df, all_embeddings, embedder, top_k=TOP_K)
    for i, result in enumerate(results, start=1):
        print(f"{i}. [{result['category']}] {result['title']} (score={result['score']:.3f})")


if __name__ == "__main__":
    main()
