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
        "shared_keywords": ["benchmark", "embedding model", "generalization"],
        "venues": ["ACL", "EMNLP", "NAACL"],
        "domain": "language understanding",
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
        "shared_keywords": ["benchmark", "feature extractor", "robustness"],
        "venues": ["CVPR", "ICCV", "MICCAI"],
        "domain": "scientific imaging",
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
        "shared_keywords": ["benchmark", "embedding model", "robustness"],
        "venues": ["ICML", "NeurIPS", "ICLR"],
        "domain": "machine learning methodology",
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
        "shared_keywords": ["benchmark", "simulation study", "generalization"],
        "venues": ["PhysRevE", "JCP", "SciPost"],
        "domain": "computational physics",
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
        "shared_keywords": ["benchmark", "uncertainty", "robustness"],
        "venues": ["JINST", "PhysRevD", "NIMA"],
        "domain": "experimental physics analysis",
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
        "shared_keywords": ["benchmark", "uncertainty", "generalization"],
        "venues": ["AISTATS", "JRSS-B", "JASA"],
        "domain": "statistical machine learning",
    },
}

GLOBAL_SHARED_TERMS = [
    "benchmark",
    "dataset",
    "generalization",
    "embedding model",
    "evaluation",
    "transfer learning",
    "uncertainty",
    "robustness",
    "ablation",
    "scientific workflow",
]

CROSS_CATEGORY_TERMS = [
    "document retrieval",
    "representation learning",
    "image segmentation",
    "question answering",
    "optimization",
    "calibration curve",
    "simulation grid",
    "causal effect",
    "event reconstruction",
    "posterior intervals",
]

NEIGHBOR_MAP = {
    "cs.CL": ["cs.LG", "stat.ML"],
    "cs.CV": ["cs.LG", "physics.data-an"],
    "cs.LG": ["cs.CL", "stat.ML"],
    "physics.comp-ph": ["physics.data-an", "cs.LG"],
    "physics.data-an": ["physics.comp-ph", "cs.CV"],
    "stat.ML": ["cs.LG", "cs.CL"],
}


def make_sample_dataset(samples_per_class: int = 167, perfect: bool = False) -> pd.DataFrame:
    records = []
    paper_id = 1
    base_year = 2020
    categories = list(CATEGORY_BLUEPRINTS.items())

    for category_index, (category, blueprint) in enumerate(categories):
        prefixes = blueprint["title_prefixes"]
        keywords = blueprint["keywords"]
        shared_keywords = blueprint["shared_keywords"]
        venues = blueprint["venues"]
        domain = blueprint["domain"]
        neighbors = NEIGHBOR_MAP[category]

        for i in range(samples_per_class):
            prefix = prefixes[i % len(prefixes)]
            keyword_a = keywords[i % len(keywords)]
            keyword_b = keywords[(i + 2) % len(keywords)]
            shared_keyword = shared_keywords[i % len(shared_keywords)]
            venue = venues[i % len(venues)]
            year = base_year + ((i + category_index) % 6)
            study_id = i + 1

            if perfect:
                title = f"{prefix} for {keyword_a.title()} Study {study_id}"
                abstract = (
                    f"We present a {category} benchmark focused on {keyword_a} and {keyword_b}. "
                    f"The study evaluates reproducible baselines, ablations, and error analysis on {venue} style tasks. "
                    f"Results emphasize category-specific signals such as {keyword_a}, {keyword_b}, and deployment constraints typical for {category} workflows."
                )
            else:
                neighbor_category = neighbors[i % len(neighbors)]
                neighbor_blueprint = CATEGORY_BLUEPRINTS[neighbor_category]
                neighbor_keyword = neighbor_blueprint["keywords"][(i + 1) % len(neighbor_blueprint["keywords"])]
                global_term = GLOBAL_SHARED_TERMS[(i + category_index) % len(GLOBAL_SHARED_TERMS)]
                cross_term = CROSS_CATEGORY_TERMS[(i * 2 + category_index) % len(CROSS_CATEGORY_TERMS)]

                title_style = i % 4
                if title_style == 0:
                    title = f"{prefix} for {domain.title()} Study {study_id}"
                elif title_style == 1:
                    title = f"{prefix}: {keyword_a.title()} and {cross_term.title()}"
                elif title_style == 2:
                    title = f"Benchmarking {shared_keyword.title()} in {domain.title()}"
                else:
                    title = f"A Comparative Study of {keyword_a.title()} for Scientific Workflows"

                abstract = (
                    f"We study {keyword_a} for {domain} with comparisons against strong baselines and shared evaluation datasets. "
                    f"The paper discusses {keyword_b}, {shared_keyword}, and {global_term} on {venue} style tasks. "
                    f"Several experiments also touch on neighboring themes such as {neighbor_keyword} and {cross_term}, making label boundaries partially overlapping rather than perfectly clean. "
                    f"Results highlight transfer, uncertainty, and practical deployment tradeoffs in scientific workflows."
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
