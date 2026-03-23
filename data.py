from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    ID_COLUMN,
    LABEL_COLUMN,
    MODEL_TEXT_COLUMN,
    PROCESSED_CSV_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TITLE_COLUMN,
)


CATEGORY_BLUEPRINTS = {
    "cs.CL": {
        "title_prefixes": [
            "Transformer Retrieval",
            "Long-Context Language Models",
            "Citation-Aware Summarization",
            "Scientific Claim Verification",
            "Terminology Grounding",
        ],
        "keywords": [
            "tokenization",
            "document retrieval",
            "question answering",
            "summarization",
            "citation context",
            "evidence ranking",
        ],
        "venues": ["ACL", "EMNLP", "NAACL"],
    },
    "cs.CV": {
        "title_prefixes": [
            "Microscopy Vision Transformers",
            "Cell Segmentation",
            "Remote Sensing Detection",
            "Radiology Image Classification",
            "Scientific Figure Parsing",
        ],
        "keywords": [
            "image segmentation",
            "lesion detection",
            "microscopy",
            "satellite imagery",
            "object detection",
            "visual backbone",
        ],
        "venues": ["CVPR", "ICCV", "MICCAI"],
    },
    "cs.LG": {
        "title_prefixes": [
            "Meta-Learning Optimizers",
            "Neural Operator Benchmarks",
            "Active Learning Loops",
            "Federated Scientific Models",
            "Probabilistic Representation Learning",
        ],
        "keywords": [
            "generalization",
            "hyperparameter search",
            "optimization",
            "representation learning",
            "uncertainty calibration",
            "benchmark suite",
        ],
        "venues": ["ICML", "NeurIPS", "ICLR"],
    },
    "physics.comp-ph": {
        "title_prefixes": [
            "Lattice Simulation Surrogates",
            "Turbulence Solvers",
            "Plasma Dynamics Emulators",
            "Quantum Monte Carlo Acceleration",
            "Inverse Problems in Physics",
        ],
        "keywords": [
            "partial differential equations",
            "simulation grid",
            "boundary conditions",
            "Hamiltonian system",
            "solver convergence",
            "numerical stability",
        ],
        "venues": ["PhysRevE", "JCP", "SciPost"],
    },
    "physics.data-an": {
        "title_prefixes": [
            "Detector Calibration Pipelines",
            "High-Energy Event Reconstruction",
            "Bayesian Measurement Uncertainty",
            "Sensor Drift Analysis",
            "Experimental Signal Denoising",
        ],
        "keywords": [
            "detector response",
            "measurement noise",
            "calibration curve",
            "event reconstruction",
            "instrument drift",
            "posterior intervals",
        ],
        "venues": ["JINST", "PhysRevD", "NIMA"],
    },
    "stat.ML": {
        "title_prefixes": [
            "Bayesian Hierarchical Models",
            "Causal Estimation under Shift",
            "Conformal Prediction for Science",
            "Experimental Design with Priors",
            "Robust Uncertainty Quantification",
        ],
        "keywords": [
            "posterior predictive checks",
            "coverage guarantees",
            "covariate shift",
            "causal effect",
            "credible intervals",
            "sampling efficiency",
        ],
        "venues": ["AISTATS", "JRSS-B", "JASA"],
    },
}


def make_sample_dataset(samples_per_class: int = 60) -> pd.DataFrame:
    records = []
    paper_id = 1
    base_year = 2020

    for category_index, (category, blueprint) in enumerate(CATEGORY_BLUEPRINTS.items()):
        prefixes = blueprint["title_prefixes"]
        keywords = blueprint["keywords"]
        venues = blueprint["venues"]

        for i in range(samples_per_class):
            prefix = prefixes[i % len(prefixes)]
            keyword_a = keywords[i % len(keywords)]
            keyword_b = keywords[(i + 2) % len(keywords)]
            venue = venues[i % len(venues)]
            year = base_year + ((i + category_index) % 6)
            study_id = i + 1

            title = f"{prefix} for {keyword_a.title()} Study {study_id}"
            abstract = (
                f"We present a {category} benchmark focused on {keyword_a} and {keyword_b}. "
                f"The study evaluates reproducible baselines, ablations, and error analysis on {venue} style tasks. "
                f"Results emphasize category-specific signals such as {keyword_a}, {keyword_b}, and deployment constraints typical for {category} workflows."
            )
            records.append(
                {
                    ID_COLUMN: paper_id,
                    TITLE_COLUMN: title,
                    TEXT_COLUMN: abstract,
                    LABEL_COLUMN: category,
                    "year": year,
                }
            )
            paper_id += 1

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
    df[MODEL_TEXT_COLUMN] = (df[TITLE_COLUMN] + ". " + df[TEXT_COLUMN]).str.strip()
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
