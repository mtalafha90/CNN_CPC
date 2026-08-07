"""The multilingual radiology-report teacher.

This is the part of the pipeline that exploits what makes this challenge
unusual: every training exam comes with a free-text report, in one of twelve
languages, written by the radiologist who saw the images.

The strategy is knowledge distillation.

* A multilingual transformer (XLM-RoBERTa by default) is fine-tuned to predict
  the findings **from the report text alone**. This is an easy task — the
  report often names the finding — so the teacher reaches a very high score.
* Its *out-of-fold* probabilities are saved. Using out-of-fold predictions is
  essential: a teacher scoring its own training data would output near-perfect
  0/1 values, which carry no more information than the labels themselves.
* The image model is then trained to match those soft probabilities alongside
  the hard labels (see :class:`rsnaknee.losses.DistillationLoss`).

Why this helps: the binary labels throw away the radiologist's hedging. A
report saying "possible small radial tear of the posterior horn" becomes a hard
1, identical to "large displaced bucket-handle tear". The teacher's soft output
preserves that gradient of certainty, and a borderline image should predict a
borderline value. Distillation gives the image model that target.

The teacher is a training-time device. If the hidden test set does supply
reports, ``--predict-test`` will also score them so you can blend the two
modalities; if it does not, the image model still carries the benefit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, load_config
from .folds import add_folds
from .metrics import evaluate, log_report
from .schema import DataSchema, discover_schema
from .utils import get_logger, read_json, seed_everything, write_json

LOGGER = get_logger()

# Column names that plausibly hold the report, tried in order.
TEXT_COLUMN_CANDIDATES = (
    "report",
    "report_text",
    "text",
    "findings",
    "impression",
    "narrative",
    "translated_report",
)


def find_text_column(frame: pd.DataFrame, preferred: str | None = None) -> str:
    """Locate the report column, preferring an explicit configuration value."""
    if preferred and preferred in frame.columns:
        return preferred
    lowered = {c.lower(): c for c in frame.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    # Fall back to the object column with the longest average content.
    object_columns = [c for c in frame.columns if frame[c].dtype == object]
    if not object_columns:
        raise ValueError("No text column found in the reports table")
    lengths = {c: frame[c].astype(str).str.len().mean() for c in object_columns}
    best = max(lengths, key=lengths.get)
    LOGGER.info("Guessed '%s' as the report column", best)
    return best


class ReportDataset(Dataset):
    """Tokenised reports with their multi-label targets."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class ReportClassifier(torch.nn.Module):
    """A multilingual encoder with a multi-label head over the pooled token."""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = AutoConfig.from_pretrained(model_name).hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        # Mean pooling over real tokens beats the CLS token for XLM-R, which is
        # not trained with a sentence-level objective.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.dropout(pooled))


def load_reports(config: Config, schema: DataSchema) -> pd.DataFrame:
    """Load the reports table and join it to the labels."""
    reports_path = config.paths.reports_csv
    if reports_path is None:
        for name in ("train_reports.csv", "reports.csv", "train_text.csv"):
            candidate = Path(config.paths.data_dir) / name
            if candidate.exists():
                reports_path = str(candidate)
                break
    train_path = config.paths.train_csv or str(Path(config.paths.data_dir) / "train.csv")
    train = pd.read_csv(train_path)
    train[schema.id_column] = train[schema.id_column].astype(str)

    if reports_path is None:
        # The reports may already be a column of train.csv.
        try:
            find_text_column(train, config.text.text_column)
            return train
        except ValueError as error:
            raise FileNotFoundError(
                "No reports CSV found and train.csv has no text column. "
                "Set paths.reports_csv."
            ) from error

    reports = pd.read_csv(reports_path)
    key = schema.id_column if schema.id_column in reports.columns else reports.columns[0]
    reports[key] = reports[key].astype(str)
    return train.merge(reports, left_on=schema.id_column, right_on=key, how="left")


def train_teacher(config: Config) -> pd.DataFrame:
    """Fine-tune the report model per fold and return out-of-fold predictions."""
    from transformers import AutoTokenizer

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_dir / "schema.json"
    schema = (
        DataSchema.from_dict(read_json(schema_path))
        if schema_path.exists()
        else discover_schema(
            config.paths.data_dir, config.paths.train_csv, config.paths.sample_submission_csv
        )
    )
    if not schema_path.exists():
        write_json(schema_path, schema.to_dict())

    frame = load_reports(config, schema)
    text_column = find_text_column(frame, config.text.text_column)
    labels = [label for label in schema.labels if label in frame.columns]
    frame[labels] = frame[labels].fillna(0).astype(np.float32)
    frame[text_column] = frame[text_column].fillna("").astype(str)

    frame = add_folds(
        frame, labels, schema.group_column, config.data.n_folds, config.data.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.text.model_name)
    oof = np.zeros((len(frame), len(labels)), dtype=np.float32)

    for fold in range(config.data.n_folds):
        seed_everything(config.data.seed + fold)
        train_mask = frame["fold"] != fold
        valid_mask = ~train_mask

        train_dataset = ReportDataset(
            frame.loc[train_mask, text_column].tolist(),
            frame.loc[train_mask, labels].to_numpy(),
            tokenizer,
            config.text.max_length,
        )
        valid_dataset = ReportDataset(
            frame.loc[valid_mask, text_column].tolist(),
            frame.loc[valid_mask, labels].to_numpy(),
            tokenizer,
            config.text.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.text.batch_size, shuffle=True, num_workers=2
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.text.batch_size * 2, shuffle=False, num_workers=2
        )

        model = ReportClassifier(config.text.model_name, len(labels)).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=config.text.learning_rate)
        total_steps = max(1, len(train_loader) * config.text.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=config.text.learning_rate, total_steps=total_steps, pct_start=0.1
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(config.text.epochs):
            model.train()
            for batch in train_loader:
                targets = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = criterion(model(**batch), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
            LOGGER.info("Teacher fold %d epoch %d done", fold, epoch)

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in valid_loader:
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions.append(torch.sigmoid(model(**batch).float()).cpu().numpy())
        oof[valid_mask.to_numpy()] = np.concatenate(predictions)
        torch.save(model.state_dict(), output_dir / f"text_fold{fold}.pt")

    report = evaluate(frame[labels].to_numpy(), oof, labels)
    log_report(report, prefix="text teacher out-of-fold:")
    write_json(output_dir / "report_text_teacher.json", report)

    result = pd.DataFrame(oof, columns=labels)
    result.insert(0, schema.id_column, frame[schema.id_column].astype(str).to_numpy())
    result.to_csv(output_dir / "text_teacher_oof.csv", index=False)
    LOGGER.info("Wrote teacher predictions to %s", output_dir / "text_teacher_oof.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multilingual report teacher")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    args = parser.parse_args()
    train_teacher(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
