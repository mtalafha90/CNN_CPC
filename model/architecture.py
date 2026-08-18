"""The working knee-MRI model: description, construction and checkpoint loading."""

from __future__ import annotations

import json

from ._implementation import (
    build_network,
    load_checkpoint,
    network_spec,
    target_names,
)

TARGETS = target_names()

WORKING_MODEL = {
    "task": "12 binary knee-MRI findings, scored as macro ROC AUC",
    "encoder": "ConvNeXt-Tiny, report-aligned then frozen; not updated during training",
    "input": "2.5D slice triplets, 16 slice positions per real series, 224x224",
    "preprocessing": "resize to 224, deterministic centred 90% crop, resize to 224",
    "series_pooling": "attention pooling over slices, plus a learned complementary summary",
    "slice_context": (
        "local depthwise slice context used during training only, "
        "bypassed exactly at inference"
    ),
    "study_context": "Transformer over the real MRI series of a study",
    "head": "12 pathology queries cross-attending to the study representation",
    "inference": "test-time slice offsets [-1, 0, 1], averaged",
    "training_endpoint": "fixed second epoch; no checkpoint selection on any labelled set",
    "expert_labels_in_gradients": 0,
}


def describe() -> dict:
    """Return the working model's architecture and training contract."""
    return dict(WORKING_MODEL)


def build(config: dict, *, normalize_input: bool = True):
    """Construct an untrained working model from a configuration mapping."""
    return build_network(network_spec(config, normalize_input=normalize_input))


def load(checkpoint, *, device: str = "cpu"):
    """Load a trained working model together with its training record.

    The architecture is read from the checkpoint, and the frozen encoder
    fingerprint recorded at training time is re-verified after the weights are
    loaded, so a checkpoint whose encoder drifted is rejected rather than
    silently scored.
    """
    return load_checkpoint(checkpoint, device=device)


def main() -> None:
    print(json.dumps({"targets": list(TARGETS), **WORKING_MODEL}, indent=2))


if __name__ == "__main__":
    main()
