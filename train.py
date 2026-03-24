from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from config import LABEL_COLUMN, MODELS_DIR, MODEL_TEXT_COLUMN, RANDOM_STATE
from models import DeepPaperClassifier, PaperClassifier


MODEL_PATH = MODELS_DIR / "paper_classifier.joblib"
DEEP_MODEL_DIR = MODELS_DIR / "paper_classifier_v2"


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_classifier(train_df: pd.DataFrame) -> PaperClassifier:
    model = PaperClassifier(text_field_name=MODEL_TEXT_COLUMN)
    model.fit(train_df[MODEL_TEXT_COLUMN].tolist(), train_df[LABEL_COLUMN].tolist())
    return model


def train_deep_classifier(
    train_df: pd.DataFrame,
    transformer_model_name: str = "allenai/scibert_scivocab_uncased",
    max_length: int = 256,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 3,
) -> DeepPaperClassifier:
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    model = DeepPaperClassifier(
        transformer_model_name=transformer_model_name,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        random_state=RANDOM_STATE,
        text_field_name=MODEL_TEXT_COLUMN,
        model_name=f"transformer_finetune:{transformer_model_name}",
    )
    labels = train_df[LABEL_COLUMN].tolist()
    model.initialize(labels)

    assert model.label_encoder is not None
    assert model.tokenizer is not None
    assert model.model is not None
    assert model.device is not None

    label_ids = model.label_encoder.transform(labels).tolist()
    dataset = TextClassificationDataset(
        texts=train_df[MODEL_TEXT_COLUMN].tolist(),
        labels=label_ids,
        tokenizer=model.tokenizer,
        max_length=model.max_length,
    )
    loader = DataLoader(dataset, batch_size=model.batch_size, shuffle=True)
    optimizer = AdamW(model.model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

    model.model.train()
    for epoch in range(model.epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{model.epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = running_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{model.epochs} mean loss: {mean_loss:.4f}")

    return model


def save_classifier(model: PaperClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path: Path = MODEL_PATH) -> PaperClassifier:
    return joblib.load(path)


def save_deep_classifier(model: DeepPaperClassifier, output_dir: Path = DEEP_MODEL_DIR) -> None:
    model.save(output_dir)


def load_deep_classifier(output_dir: Path = DEEP_MODEL_DIR) -> DeepPaperClassifier:
    return DeepPaperClassifier.load(output_dir)
