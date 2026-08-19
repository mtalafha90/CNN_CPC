"""Let the last part of the frozen encoder learn.

Every experiment since B17 has trained the head on top of a completely frozen
encoder. That is 27.8M parameters held fixed against 18.9M that learn, so only
about 40% of the model has ever adapted to knee MRI -- and the eight most
recent experiments, all confined to that 40%, moved the score by roughly +0.015
with every interval crossing zero.

This unfreezes the encoder from the output end. The early layers see edges and
textures, which transfer from natural images perfectly well; the late layers
carry the object-level meaning, which is where knee MRI differs most from the
photographs the encoder learned on. Unfreezing from the end therefore adapts
what needs adapting and leaves alone what does not.

Two deliberate restraints:

- The encoder stays in eval mode. ConvNeXt normalises with LayerNorm, which
  behaves identically either way, but eval also disables stochastic depth. So
  the forward pass keeps exactly the shape it had, and only the flow of
  gradients changes -- the comparison against a frozen run stays clean.
- The unfrozen parameters get their own, much smaller learning rate. Pretrained
  features are worth more than a randomly initialised head and are easily
  destroyed by the head's step size.
"""
from __future__ import annotations

from torch import nn

# The encoder's parts, listed from the output backwards. Unfreezing `n` stages
# takes the first `n` of these. `pre_classifier` is the final normalisation and
# is tiny; it belongs with the last stage rather than on its own.
ENCODER_BLOCKS_FROM_OUTPUT = (
    ("pre_classifier", "features.7"),   # last stage plus its output norm
    ("features.6",),                    # the downsample feeding it
    ("features.5",),                    # third stage
    ("features.4",),
    ("features.3",),
)

MAX_TRAINABLE_STAGES = len(ENCODER_BLOCKS_FROM_OUTPUT)

# Pretrained features need a far gentler step than a fresh head.
DEFAULT_ENCODER_LR_SCALE = 0.05


def _prefixes_for(stages: int) -> tuple[str, ...]:
    if not 0 <= stages <= MAX_TRAINABLE_STAGES:
        raise ValueError(
            f"encoder_trainable_stages must be between 0 and {MAX_TRAINABLE_STAGES}"
        )
    names: list[str] = []
    for block in ENCODER_BLOCKS_FROM_OUTPUT[:stages]:
        names.extend(block)
    return tuple(names)


def unfreeze_encoder_tail(model: nn.Module, stages: int) -> dict:
    """Free the last `stages` encoder blocks; everything else stays frozen.

    Call after the usual freeze, so the default remains a fully frozen encoder
    and this only ever relaxes it.
    """
    prefixes = _prefixes_for(stages)
    if not prefixes:
        return {
            "encoder_trainable_stages": 0,
            "encoder_trainable_parameters": 0,
            "encoder_trainable_prefixes": [],
        }

    freed = 0
    for name, parameter in model.encoder.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            parameter.requires_grad_(True)
            freed += parameter.numel()

    if freed == 0:
        raise RuntimeError(
            f"no encoder parameters matched {prefixes}; the encoder layout changed"
        )

    # The encoder stays in eval mode on purpose: the forward pass is unchanged,
    # only gradients now flow.
    model.encoder.eval()
    return {
        "encoder_trainable_stages": int(stages),
        "encoder_trainable_parameters": int(freed),
        "encoder_trainable_prefixes": list(prefixes),
    }


def split_parameters(model: nn.Module) -> tuple[list, list]:
    """Separate trainable head parameters from trainable encoder parameters."""
    head, encoder = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if name.startswith("encoder.") else head).append(parameter)
    return head, encoder


def parameter_groups(model: nn.Module, *, head_lr: float, encoder_lr_scale: float) -> list[dict]:
    """Build optimiser groups, giving the encoder a gentler learning rate."""
    if not 0.0 < encoder_lr_scale <= 1.0:
        raise ValueError("encoder_lr_scale must be in (0, 1]")
    head, encoder = split_parameters(model)
    if not head:
        raise RuntimeError("no trainable head parameters")

    groups = [{"params": head, "lr": float(head_lr), "name": "head"}]
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": float(head_lr) * float(encoder_lr_scale),
                "name": "encoder",
            }
        )
    return groups
