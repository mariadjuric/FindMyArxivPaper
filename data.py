from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    ID_COLUMN,
    LABEL_COLUMN,
    PROCESSED_CSV_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TITLE_COLUMN,
)


def make_sample_dataset() -> pd.DataFrame:
    records = [
        {"paper_id": 1, "title": "Transformers for Scientific Claim Verification", "abstract": "We study transformer encoders for verifying scientific claims against evidence passages and measure retrieval and entailment quality.", "category": "cs.CL", "year": 2024},
        {"paper_id": 2, "title": "Graph Neural Networks for Molecule Property Prediction", "abstract": "This work applies graph neural networks to molecular property prediction with uncertainty estimates and benchmark comparisons.", "category": "cs.LG", "year": 2023},
        {"paper_id": 3, "title": "Vision Transformers for Medical Image Segmentation", "abstract": "We propose a vision transformer model for semantic segmentation in medical imaging and compare against CNN baselines.", "category": "cs.CV", "year": 2024},
        {"paper_id": 4, "title": "Contrastive Learning for Scientific Abstract Retrieval", "abstract": "Contrastive pretraining improves semantic search over scientific abstracts by learning domain-specific text representations.", "category": "cs.CL", "year": 2022},
        {"paper_id": 5, "title": "Diffusion Models for Inverse Problems", "abstract": "We investigate diffusion-based generative priors for solving inverse problems arising in imaging and physics.", "category": "physics.comp-ph", "year": 2024},
        {"paper_id": 6, "title": "Self-Supervised Representation Learning for Remote Sensing", "abstract": "Self-supervised methods are evaluated for remote sensing image understanding with limited labels.", "category": "cs.CV", "year": 2023},
        {"paper_id": 7, "title": "Large Language Models for Literature Review Assistance", "abstract": "We analyze large language models for summarizing and organizing related work in scientific domains.", "category": "cs.CL", "year": 2024},
        {"paper_id": 8, "title": "Bayesian Neural Networks for Experimental Physics", "abstract": "Bayesian neural networks are used to model uncertainty in experimental measurements and simulation-to-data transfer.", "category": "physics.data-an", "year": 2021},
        {"paper_id": 9, "title": "Multimodal Learning for Document Figure Understanding", "abstract": "This paper studies multimodal encoders that jointly model scientific figures and captions for document understanding.", "category": "cs.CV", "year": 2024},
        {"paper_id": 10, "title": "Topic Modeling of arXiv Abstracts", "abstract": "We compare modern topic discovery pipelines for organizing arXiv abstracts across machine learning and physics.", "category": "cs.CL", "year": 2022},
        {"paper_id": 11, "title": "Neural Operators for PDE Surrogates", "abstract": "Neural operator models are evaluated as surrogate models for partial differential equations in scientific computing.", "category": "cs.LG", "year": 2023},
        {"paper_id": 12, "title": "Few-Shot Visual Recognition in Scientific Images", "abstract": "We benchmark few-shot learning methods on scientific image classification tasks with scarce labels.", "category": "cs.CV", "year": 2022}
    ]
    return pd.DataFrame(records)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = make_sample_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return preprocess_dataset(df)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = [TITLE_COLUMN, TEXT_COLUMN, LABEL_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = range(1, len(df) + 1)

    df[TITLE_COLUMN] = df[TITLE_COLUMN].fillna("").astype(str).str.strip()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("unknown").astype(str).str.strip()
    df = df[df[TITLE_COLUMN] != ""]
    df = df[df[TEXT_COLUMN] != ""]
    df = df.reset_index(drop=True)
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    return df


def make_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = df[LABEL_COLUMN].value_counts()
    can_stratify = df[LABEL_COLUMN].nunique() > 1 and (label_counts >= 2).all()
    stratify = df[LABEL_COLUMN] if can_stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
