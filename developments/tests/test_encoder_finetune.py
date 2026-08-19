"""Freeing the encoder's tail must free exactly that, and nothing else.

Every experiment since B17 trained on a fully frozen encoder, so the default
has to stay frozen and the relaxation has to be precise: the freed parameters
must actually receive gradient, the rest must stay untouched, and pretrained
weights must not be stepped at the head's learning rate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.encoder_finetune import (  # noqa: E402
    MAX_TRAINABLE_STAGES,
    parameter_groups,
    split_parameters,
    unfreeze_encoder_tail,
)
from rsna_knee.model import ConvNeXtSliceEncoder  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNeXtSliceEncoder(pretrained_weights=False, normalize_input=True)
        self.head = torch.nn.Linear(self.encoder.out_dim, 12)

    def forward(self, x):
        return self.head(self.encoder(x))


def _frozen_model():
    model = _Model()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model


def test_zero_stages_changes_nothing():
    """The historical behaviour must survive as the default."""
    model = _frozen_model()
    info = unfreeze_encoder_tail(model, 0)
    assert info["encoder_trainable_parameters"] == 0
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_one_stage_frees_the_output_end_only():
    model = _frozen_model()
    unfreeze_encoder_tail(model, 1)
    freed = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
    assert freed, "nothing was freed"
    assert all(n.startswith(("pre_classifier", "features.7")) for n in freed)
    # the early layers, which transfer fine from natural images, stay fixed
    assert not any(n.startswith("features.0") for n in freed)


def test_more_stages_free_strictly_more():
    counts = []
    for stages in range(MAX_TRAINABLE_STAGES + 1):
        model = _frozen_model()
        counts.append(unfreeze_encoder_tail(model, stages)["encoder_trainable_parameters"])
    assert counts == sorted(counts)
    assert counts[0] == 0 and counts[-1] > counts[1]


@pytest.mark.parametrize("stages", [-1, MAX_TRAINABLE_STAGES + 1])
def test_an_impossible_stage_count_is_refused(stages):
    with pytest.raises(ValueError, match="encoder_trainable_stages"):
        unfreeze_encoder_tail(_frozen_model(), stages)


def test_the_encoder_stays_in_eval_mode():
    """Keeping eval disables stochastic depth, so the forward pass is unchanged."""
    model = _frozen_model()
    unfreeze_encoder_tail(model, 1)
    model.train()
    unfreeze_encoder_tail(model, 1)
    assert not model.encoder.training


def test_the_freed_parameters_actually_receive_gradient():
    model = _frozen_model()
    unfreeze_encoder_tail(model, 1)
    model(torch.randn(2, 3, 64, 64)).sum().backward()

    _, encoder = split_parameters(model)
    assert encoder
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder)
    # and the frozen majority receives none
    assert all(
        p.grad is None for p in model.encoder.parameters() if not p.requires_grad
    )


def test_the_encoder_gets_a_gentler_learning_rate():
    """Pretrained features are easily destroyed by the head's step size."""
    model = _frozen_model()
    unfreeze_encoder_tail(model, 1)
    groups = parameter_groups(model, head_lr=1e-3, encoder_lr_scale=0.05)
    by_name = {g["name"]: g["lr"] for g in groups}
    assert by_name["head"] == pytest.approx(1e-3)
    assert by_name["encoder"] == pytest.approx(5e-5)


def test_a_frozen_encoder_produces_a_single_group():
    groups = parameter_groups(_frozen_model(), head_lr=1e-3, encoder_lr_scale=0.05)
    assert [g["name"] for g in groups] == ["head"]


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
def test_an_impossible_learning_rate_scale_is_refused(scale):
    model = _frozen_model()
    unfreeze_encoder_tail(model, 1)
    with pytest.raises(ValueError, match="encoder_lr_scale"):
        parameter_groups(model, head_lr=1e-3, encoder_lr_scale=scale)
