from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import DUAL_STREAMS
from .data import backfill_series_metadata, build_series_index, load_series_csv
from .dicom import find_series_dir, preprocess_triplets, read_dicom_series


@dataclass
class PreflightResult:
    split: str
    studies_sampled: int
    streams_possible: int
    streams_selected: int
    streams_missing: int
    directories_found: int
    streams_decoded: int
    candidate_files: int
    file_decode_failures: int
    decoded_frames: int
    metadata_fields_missing: int
    metadata_fields_repaired: int
    decode_failure_rate: float
    file_decode_failure_rate: float
    missing_stream_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"preflight split={self.split} studies={self.studies_sampled} "
            f"selected={self.streams_selected}/{self.streams_possible} "
            f"decoded={self.streams_decoded}/{self.streams_selected} "
            f"stream_decode_failure_rate={self.decode_failure_rate:.1%} "
            f"file_decode_failure_rate={self.file_decode_failure_rate:.1%} "
            f"missing_stream_rate={self.missing_stream_rate:.1%} "
            f"metadata_fields_repaired={self.metadata_fields_repaired}/{self.metadata_fields_missing}"
        )


def run_preflight(
    data_root: str | Path,
    *,
    split: str = "train",
    series_csv: str | Path | None = None,
    study_uids: list[str] | None = None,
    sample_size: int = 24,
    stream_mode: str = "dual",
    image_size: int = 96,
    triplet_gap: int = 1,
    seed: int = 2026,
    max_decode_failure_rate: float = 0.05,
    max_file_decode_failure_rate: float = 0.05,
    strict: bool = True,
) -> PreflightResult:
    if sample_size < 1 or image_size < 1 or triplet_gap < 1:
        raise ValueError("sample_size, image_size and triplet_gap must be positive")
    if not 0 <= max_decode_failure_rate <= 1 or not 0 <= max_file_decode_failure_rate <= 1:
        raise ValueError("decode failure rates must be in [0,1]")
    if stream_mode not in {"best", "dual"}:
        raise ValueError("stream_mode must be best or dual")

    root = Path(data_root)
    csv_path = Path(series_csv) if series_csv else root / f"{split}_series.csv"
    series = load_series_csv(csv_path)
    series, repair = backfill_series_metadata(series, root, split=split)

    available = sorted(series["StudyInstanceUID"].unique().tolist())
    if study_uids is not None:
        wanted = set(map(str, study_uids))
        available = [uid for uid in available if uid in wanted]
    if not available:
        raise RuntimeError(f"preflight found no studies for split={split}")

    rng = np.random.default_rng(seed)
    chosen = (
        sorted(rng.choice(available, size=sample_size, replace=False).tolist())
        if len(available) > sample_size
        else available
    )
    index = build_series_index(series, chosen, stream_mode)
    stream_names = list(DUAL_STREAMS) if stream_mode == "dual" else ["sagittal", "coronal", "axial"]
    possible = len(chosen) * len(stream_names)
    selected = found = decoded = candidate_files = file_failures = frames = 0

    for uid in chosen:
        for stream_name in stream_names:
            series_uid = index.get(uid, {}).get(stream_name)
            if not series_uid:
                continue
            selected += 1
            path = find_series_dir(root, split, uid, series_uid)
            if path is None:
                continue
            found += 1
            try:
                volume, stats = read_dicom_series(path, return_stats=True)
                tensor = preprocess_triplets(
                    volume,
                    n_slices=min(4, max(1, len(volume))),
                    image_size=image_size,
                    gap=triplet_gap,
                )
                if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                    raise RuntimeError("non-finite or empty preprocessed tensor")
                decoded += 1
                candidate_files += int(stats["candidate_files"])
                file_failures += int(stats["decode_failures"])
                frames += int(stats["decoded_frames"])
            except Exception:
                continue

    missing = possible - selected
    metadata_fields_missing = int(
        repair["missing_plane"] + repair["missing_fluid"] + repair["missing_fat_suppression"]
    )
    metadata_fields_repaired = int(
        repair["repaired_plane"] + repair["repaired_fluid"] + repair["repaired_fat_suppression"]
    )
    stream_failure_rate = float(1.0 - decoded / max(selected, 1))
    file_failure_rate = float(file_failures / max(candidate_files, 1))
    result = PreflightResult(
        split=split,
        studies_sampled=len(chosen),
        streams_possible=possible,
        streams_selected=selected,
        streams_missing=missing,
        directories_found=found,
        streams_decoded=decoded,
        candidate_files=candidate_files,
        file_decode_failures=file_failures,
        decoded_frames=frames,
        metadata_fields_missing=metadata_fields_missing,
        metadata_fields_repaired=metadata_fields_repaired,
        decode_failure_rate=stream_failure_rate,
        file_decode_failure_rate=file_failure_rate,
        missing_stream_rate=float(missing / max(possible, 1)),
    )
    if selected == 0:
        raise RuntimeError(result.summary() + "; no MRI streams were selectable")
    if strict and stream_failure_rate > max_decode_failure_rate:
        raise RuntimeError(
            result.summary()
            + f" exceeds max_decode_failure_rate={max_decode_failure_rate:.1%}; fix DICOM/path/codec issues"
        )
    if strict and file_failure_rate > max_file_decode_failure_rate:
        raise RuntimeError(
            result.summary()
            + f" exceeds max_file_decode_failure_rate={max_file_decode_failure_rate:.1%}; "
            "partial series corruption must be resolved before training"
        )
    return result
