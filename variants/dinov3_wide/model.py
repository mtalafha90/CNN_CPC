"""Build the working model's head at whatever width the encoder emits.

The head is already width-agnostic: `HierarchicalSeriesKneeMILNet` reads
`self.encoder.out_dim` and sizes the study Transformer, the pathology queries
and the output projection from it. What it is not is *encoder*-agnostic -- it
constructs `ConvNeXtSliceEncoder` directly.

So a 1024-d encoder needs no projection layer bolted on top. It only needs the
head to be built around a different encoder in the first place. This module
substitutes the encoder class for the duration of construction and then hands
back an ordinary model.

That substitution is the whole trick, and it is why this lives in a variant
directory: it reaches into a frozen module's namespace, which is acceptable for
an experiment kept separate and unacceptable in the supported path.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from model._implementation import ensure_developments_source

from .encoder import DINOV3_MODELS, WideDinoV3SliceEncoder

ensure_developments_source()

from rsna_knee import b12_1_hierarchical  # noqa: E402
from rsna_knee.b17_training import encoder_state_sha256, freeze_encoder  # noqa: E402
from rsna_knee.b34_training_only_context_scaffold import (  # noqa: E402
    b34_model_spec,
    build_b34_model,
)

WIDE_ARCHITECTURE_SUFFIX = "_wide_dinov3"


# The frozen contracts assert exact parameter counts, written as literals
# derived from a 768-d encoder: the complementary query and gate are one vector
# each (d), and the depthwise k=3 context is 3d. Their intent is "exactly these
# shapes for this width", so a wider encoder needs the same expressions
# evaluated at its own d, not the 768 answers.
_WIDTH_SCALED = {
    "QUERY_PARAMETERS": lambda d: d,
    "GATE_PARAMETERS": lambda d: d,
    "CONTEXT_PARAMETERS": lambda d: 3 * d,
}
_CONTRACT_MODULES = (
    "b29_complementary_series_pool",
    "b31_local_context_complementary_pool",
    "b34_training_only_context_scaffold",
)


def _scaled_counts(width: int) -> dict[str, int]:
    """The parameter counts the frozen contracts should assert at this width."""
    return {
        "query": width,
        "gate": width,
        "context": 3 * width,
        "b29_new": 2 * width,
        "b31_new": 5 * width,
    }


@contextlib.contextmanager
def _contracts_scaled_to(width: int):
    """Evaluate the frozen parameter-count contracts at a different width."""
    import importlib

    counts = _scaled_counts(width)
    saved: list[tuple[object, str, int]] = []
    for module_name in _CONTRACT_MODULES:
        module = importlib.import_module(f"rsna_knee.{module_name}")
        for attribute in dir(module):
            if not attribute.endswith("_PARAMETERS"):
                continue
            current = getattr(module, attribute)
            if not isinstance(current, int):
                continue
            saved.append((module, attribute, current))
            if attribute.endswith("_NEW_PARAMETERS"):
                new = counts["b29_new"] if attribute.startswith("B29") else counts["b31_new"]
            else:
                suffix = attribute.split("_", 2)[-1]
                scale = _WIDTH_SCALED.get(suffix)
                new = scale(width) if scale else current
            setattr(module, attribute, new)
    try:
        yield counts
    finally:
        for module, attribute, original in saved:
            setattr(module, attribute, original)


@contextlib.contextmanager
def _encoder_substituted(variant: str, *, pretrained_weights: bool):
    """Make the head construct a DINOv3 encoder instead of the ConvNeXt one."""
    original = b12_1_hierarchical.ConvNeXtSliceEncoder

    def factory(in_channels, *, pretrained_weights=False, normalize_input=True):
        # The head passes its own pretrained_weights through; ignore it and use
        # the caller's intent, since ImageNet weights are meaningless here.
        return WideDinoV3SliceEncoder(
            in_channels,
            variant=variant,
            pretrained_weights=_encoder_substituted.pretrained,
            normalize_input=normalize_input,
        )

    _encoder_substituted.pretrained = bool(pretrained_weights)
    b12_1_hierarchical.ConvNeXtSliceEncoder = factory
    try:
        yield
    finally:
        b12_1_hierarchical.ConvNeXtSliceEncoder = original


def wide_model_spec(config: dict, variant: str, *, normalize_input: bool = True) -> dict:
    """The frozen model spec, marked as a wide-encoder variant."""
    if variant not in DINOV3_MODELS:
        known = ", ".join(sorted(DINOV3_MODELS))
        raise ValueError(f"unknown DINOv3 variant {variant!r}; known: {known}")
    spec = dict(b34_model_spec(config, normalize_input=normalize_input))
    width = DINOV3_MODELS[variant][1]
    counts = _scaled_counts(width)

    spec["encoder_source"] = "dinov3"
    spec["dinov3_variant"] = variant
    spec["dinov3_model_name"] = DINOV3_MODELS[variant][0]
    spec["encoder_width"] = width
    spec["variant_architecture"] = spec["architecture"] + WIDE_ARCHITECTURE_SUFFIX

    # The spec's declared counts are checked against the contracts, so they have
    # to be evaluated at this width too.
    spec["b29_new_parameter_count"] = counts["b29_new"]
    spec["b31_context_parameter_count"] = counts["context"]
    spec["b31_new_parameter_count"] = counts["b31_new"]
    spec["b34_new_parameter_count"] = counts["b31_new"]
    return spec


def build_wide_model(spec: dict, *, pretrained_weights: bool = True):
    """Build the frozen head sized to a DINOv3 encoder of the spec's width."""
    variant = str(spec.get("dinov3_variant", ""))
    if variant not in DINOV3_MODELS:
        raise ValueError("spec does not name a known DINOv3 variant")

    # build_b34_model asserts the frozen architecture string, so hand it the
    # untouched spec and keep the variant marker alongside.
    width = int(spec["encoder_width"])
    frozen_spec = {k: v for k, v in spec.items() if k != "variant_architecture"}
    with _contracts_scaled_to(width):
        with _encoder_substituted(variant, pretrained_weights=pretrained_weights):
            model = build_b34_model(frozen_spec, pretrained_weights=False)

    expected = int(spec["encoder_width"])
    if int(model.encoder.out_dim) != expected:
        raise RuntimeError(
            f"encoder width {model.encoder.out_dim} does not match the spec's {expected}"
        )
    return model


def save_wide_checkpoint(model, spec: dict, path: str | Path, **extra) -> Path:
    """Write a checkpoint that records the width it needs to be rebuilt at."""
    freeze_encoder(model)
    payload = {
        "model_spec": spec,
        "model_state": model.state_dict(),
        "encoder_sha256_final": encoder_state_sha256(model.encoder),
        "encoder_frozen": True,
        "encoder": model.encoder.describe(),
        **extra,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def load_wide_checkpoint(path: str | Path, *, device: str = "cpu"):
    """Rebuild a wide-encoder model from its own checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("model_spec", "model_state"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key!r}")

    model = build_wide_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)

    recorded = str(payload.get("encoder_sha256_final", ""))
    if recorded and encoder_state_sha256(model.encoder) != recorded:
        raise ValueError("reconstructed encoder fingerprint does not match the checkpoint")
    return model.to(device), payload
