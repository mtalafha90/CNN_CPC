"""Inference and submission writing.

The competition awards a separate prize for efficient models, so this script
reports its own wall-clock time and keeps the expensive knobs optional. In
rough order of cost per unit of gain:

* ensembling folds — the largest gain, and the largest cost;
* slice-shift test-time augmentation — cheap, small but reliable gain;
* half precision — free speed on any modern GPU, no measurable accuracy cost.

For an efficiency-track submission, run a single fold at ``--folds 0`` with
``inference.tta_slice_shift=false``; for the accuracy leaderboard, use every
fold.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .dataset import DatasetConfig, KneeExamDataset, collate_exams
from .preprocess import build_cache
from .schema import DataSchema, write_submission
from .train import amp_dtype_from_string, build_model
from .utils import get_logger

LOGGER = get_logger()


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, DataSchema, Config]:
    """Rebuild a model from a training checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    config = Config()
    from .config import _merge

    _merge(config, state["config"])
    schema = DataSchema.from_dict(state["schema"])
    model = build_model(config, schema.num_labels)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, schema, config


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    tta_slice_shift: bool = False,
    tta_hflip: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Predict over a loader with optional test-time augmentation."""
    probabilities: list[np.ndarray] = []
    exam_ids: list[str] = []

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        series_type = batch["series_type"].to(device, non_blocking=True)
        series_mask = batch["series_mask"].to(device, non_blocking=True)

        views = [pixels]
        if tta_slice_shift and pixels.shape[2] > 2:
            # Roll the slice axis by one: a different but equally valid sampling
            # of the same anatomy.
            views.append(torch.roll(pixels, shifts=1, dims=2))
        if tta_hflip:
            views.append(torch.flip(pixels, dims=[-1]))

        accumulated = None
        for view in views:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(view, series_type, series_mask)["logits"]
            probability = torch.sigmoid(logits.float())
            accumulated = probability if accumulated is None else accumulated + probability

        probabilities.append((accumulated / len(views)).cpu().numpy())
        exam_ids.extend(batch["exam_id"])

    return np.concatenate(probabilities), exam_ids


def run_inference(config: Config, fold_indices: list[int] | None, output_csv: str) -> pd.DataFrame:
    """Ensemble the requested folds over the test set and write a submission."""
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.paths.output_dir)

    checkpoints = sorted(output_dir.glob("fold*.pt"))
    if fold_indices is not None:
        checkpoints = [p for p in checkpoints if int(p.stem.replace("fold", "")) in fold_indices]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching fold*.pt in {output_dir}")
    LOGGER.info("Ensembling %d checkpoints", len(checkpoints))

    cache_dir = Path(config.paths.cache_dir)
    manifest_path = cache_dir / "series_manifest.csv"
    if config.paths.test_dicom_dir and not manifest_path.exists():
        LOGGER.info("Building the test cache from %s", config.paths.test_dicom_dir)
        build_cache(
            config.paths.test_dicom_dir,
            cache_dir,
            size=max(config.data.image_size, 256),
            max_slices=max(config.data.depth * 2, 48),
            workers=config.data.num_workers,
        )
    manifest = pd.read_csv(manifest_path)
    manifest["exam_id"] = manifest["exam_id"].astype(str)

    _, schema, _ = load_checkpoint(checkpoints[0], device)
    exam_frame = pd.DataFrame({schema.id_column: sorted(manifest["exam_id"].unique())})

    dataset = KneeExamDataset(
        exam_frame,
        manifest,
        DatasetConfig(
            cache_dir=str(cache_dir),
            image_size=config.data.image_size,
            depth=config.data.depth,
            max_series=config.data.max_series,
            augment=False,
            series_dropout=0.0,
            random_erase=0.0,
        ),
        schema.id_column,
        label_columns=None,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_exams,
        pin_memory=True,
    )

    amp_dtype = (
        torch.float16
        if config.inference.half and device.type == "cuda"
        else amp_dtype_from_string(config.train.amp_dtype)
    )

    total: np.ndarray | None = None
    exam_ids: list[str] = []
    for checkpoint in checkpoints:
        model, _, _ = load_checkpoint(checkpoint, device)
        predictions, exam_ids = predict_loader(
            model,
            loader,
            device,
            amp_dtype,
            config.inference.tta_slice_shift,
            config.inference.tta_hflip,
        )
        total = predictions if total is None else total + predictions
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert total is not None
    predictions = total / len(checkpoints)

    frame = write_submission(
        predictions,
        exam_ids,
        schema,
        output_csv,
        config.paths.sample_submission_csv,
    )
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Inference finished in %.1f s for %d exams (%.2f s per exam)",
        elapsed,
        len(exam_ids),
        elapsed / max(1, len(exam_ids)),
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and write a submission")
    parser.add_argument("--config", default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    run_inference(config, args.folds, args.output)


if __name__ == "__main__":
    main()
