from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .data import backfill_series_metadata, build_series_index, load_series_csv
from .dicom import find_series_dir, preprocess_volume, read_dicom_series


@dataclass
class PreflightResult:
    split: str
    studies_sampled: int
    streams_expected: int
    streams_selected: int
    directories_found: int
    streams_decoded: int
    candidate_files: int
    decode_failures: int
    decoded_frames: int
    metadata_missing: int
    metadata_repaired: int
    failure_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"preflight split={self.split} studies={self.studies_sampled} "
            f"decoded={self.streams_decoded}/{self.streams_expected} "
            f"failure_rate={self.failure_rate:.1%} "
            f"metadata_repaired={self.metadata_repaired}/{self.metadata_missing} "
            f"decode_failures={self.decode_failures}"
        )


def run_preflight(
    data_root: str | Path,
    *,
    split: str = "train",
    series_csv: str | Path | None = None,
    study_uids: list[str] | None = None,
    sample_size: int = 24,
    stream_mode: str = "best",
    image_size: int = 96,
    seed: int = 2026,
    max_failure_rate: float = 0.05,
    strict: bool = True,
) -> PreflightResult:
    """Audit a representative sample of real competition DICOM streams.

    This intentionally performs actual pixel decoding. It catches missing codec
    support, path/layout errors, unreadable series and metadata gaps before a GPU
    training session is allowed to consume hours while silently seeing zeros.
    """
    root = Path(data_root)
    csv_path = Path(series_csv) if series_csv else root / f"{split}_series.csv"
    series = load_series_csv(csv_path)
    series, repair = backfill_series_metadata(series, root, split=split)

    available = sorted(series["StudyInstanceUID"].astype(str).unique().tolist())
    if study_uids is not None:
        wanted = set(map(str, study_uids))
        available = [uid for uid in available if uid in wanted]
    if not available:
        raise RuntimeError(f"preflight found no studies for split={split}")

    rng = np.random.default_rng(seed)
    if len(available) > sample_size:
        chosen = sorted(rng.choice(available, size=sample_size, replace=False).tolist())
    else:
        chosen = available

    index = build_series_index(series, chosen, stream_mode)
    stream_names = sorted(next(iter(index.values())).keys()) if index else []
    expected = len(chosen) * len(stream_names)
    selected = found = decoded = candidate_files = failures = frames = 0

    for uid in chosen:
        for stream in stream_names:
            series_uid = index.get(uid, {}).get(stream)
            if not series_uid:
                continue
            selected += 1
            path = find_series_dir(root, split, uid, str(series_uid))
            if path is None:
                continue
            found += 1
            try:
                volume, stats = read_dicom_series(path, return_stats=True)
                tensor = preprocess_volume(volume, n_slices=min(4, len(volume)), image_size=image_size)
                if tensor.numel() == 0 or not np.isfinite(tensor.numpy()).all():
                    raise RuntimeError("non-finite or empty preprocessed tensor")
                decoded += 1
                candidate_files += int(stats["candidate_files"])
                failures += int(stats["decode_failures"])
                frames += int(stats["decoded_frames"])
            except Exception:
                # Count the stream as failed. File-level stats may be unavailable
                # when the entire series fails, which is why stream failure rate
                # is the gating statistic.
                continue

    failure_rate = 1.0 - decoded / max(expected, 1)
    result = PreflightResult(
        split=split,
        studies_sampled=len(chosen),
        streams_expected=expected,
        streams_selected=selected,
        directories_found=found,
        streams_decoded=decoded,
        candidate_files=candidate_files,
        decode_failures=failures,
        decoded_frames=frames,
        metadata_missing=int(repair["missing"]),
        metadata_repaired=int(repair["repaired"]),
        failure_rate=float(failure_rate),
    )
    if strict and failure_rate > max_failure_rate:
        raise RuntimeError(
            result.summary()
            + f" exceeds max_failure_rate={max_failure_rate:.1%}; fix DICOM/path/metadata issues first"
        )
    return result
