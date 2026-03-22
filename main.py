from __future__ import annotations

import numpy as np

from config import EMBEDDER_NAME, TOP_K, ensure_directories
from data import load_dataset, make_splits
from evaluate import evaluate_classification, evaluate_retrieval
from models import PaperEmbedder
from plots import plot_confusion_matrix, plot_embedding_projection, plot_label_distribution
from search import semantic_search
from train import save_classifier, train_classifier
from utils import print_section


def main() -> None:
    ensure_directories()

    print_section("Loading data")
    df = load_dataset()
    print(f"Loaded {len(df)} papers")

    print_section("Creating splits")
    train_df, test_df = make_splits(df)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    print_section("Building embeddings")
    embedder = PaperEmbedder(EMBEDDER_NAME)
    all_embeddings = embedder.encode(df["abstract"].tolist())
    train_embeddings = embedder.encode(train_df["abstract"].tolist())
    test_embeddings = embedder.encode(test_df["abstract"].tolist())
    print(f"Embedding shape: {all_embeddings.shape}")

    print_section("Training classifier")
    classifier = train_classifier(train_df, train_embeddings)
    save_classifier(classifier)
    print("Saved classifier to outputs/models/")

    print_section("Evaluating classifier")
    clf_metrics = evaluate_classification(classifier, test_df, test_embeddings)
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

    print_section("Example semantic search")
    query = "transformer models for scientific document retrieval"
    results = semantic_search(query, df, all_embeddings, embedder, top_k=TOP_K)
    for i, result in enumerate(results, start=1):
        print(f"{i}. [{result['category']}] {result['title']} (score={result['score']:.3f})")


if __name__ == "__main__":
    main()
