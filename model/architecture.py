"""Architecture description and checkpoint loader for the active working model."""

from __future__ import annotations

import json
from pathlib import Path

from .bootstrap import ensure_developments_source

TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

CURRENT_MODEL = {
    "name": "B20_crop_only_joint_focus",
    "checkpoint": "runs/b20_crop_focus/b20_model.pt",
    "canonical_epoch": 2,
    "encoder": "frozen ConvNeXt-Tiny, competition-adapted through B15/B16",
    "input": "2.5D MRI slice triplets, 16 slices per real series, 224x224",
    "preprocessing": "resize 224 -> centered 90% crop -> resize 224",
    "series_aggregation": "learned per-series attention pooling",
    "study_context": "Transformer over real MRI series tokens",
    "outputs": "12 pathology-query logits",
    "status": "ACTIVE WORKING MODEL",
}


def load_current_model(checkpoint: str | Path, *, device: str = "cpu"):
    """Load and verify a selected B20 checkpoint."""
    ensure_developments_source()
    from rsna_knee.b20_crop_focus import load_b20_checkpoint

    return load_b20_checkpoint(checkpoint, device=device)


def main() -> None:
    print(json.dumps(CURRENT_MODEL, indent=2))


if __name__ == "__main__":
    main()
