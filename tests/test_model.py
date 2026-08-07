import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.constants import TARGETS
from rsna_knee.dicom import preprocess_triplets
from rsna_knee.model import KneeMILNet
from rsna_knee.report_labels import combine_gold_and_pseudo
from rsna_knee.training import confidence_gated_ranking_loss, macro_weighted_bce


def test_preprocess_triplets_shape_and_range():
    volume = np.arange(9 * 12 * 10, dtype=np.float32).reshape(9, 12, 10)
    out = preprocess_triplets(volume, n_slices=5, image_size=16, gap=1)
    assert out.shape == (5, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def _small_model(n_slices=2):
    return KneeMILNet(
        n_streams=6,
        n_slices=n_slices,
        pretrained_weights=False,
        normalize_input=False,
        encoder_batch_size=2,
        gradient_checkpointing=False,
        transformer_layers=1,
        transformer_heads=8,
        pathology_layers=1,
    ).eval()


def test_transformer_pathology_model_forward_is_finite():
    model = _small_model(2)
    x = torch.rand(1, 6, 2, 3, 64, 64)
    present = torch.tensor([[1, 1, 1, 0, 1, 0]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x, present)
    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_model_allows_all_streams_missing_without_nan():
    model = _small_model(2)
    with torch.no_grad():
        logits = model(torch.zeros(1, 6, 2, 3, 64, 64), torch.zeros(1, 6))
    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_model_rejects_wrong_slice_count():
    model = _small_model(4)
    with pytest.raises(ValueError, match="sampled slices"):
        model(torch.rand(1, 6, 3, 3, 64, 64), torch.ones(1, 6))


def test_partial_gold_overrides_only_annotated_cells():
    df = pd.DataFrame({"StudyInstanceUID": ["x"], "Report": [""]})
    for target in TARGETS:
        df[target] = np.nan
    df.loc[0, "ACL"] = 1.0
    pseudo = np.full((1, len(TARGETS)), 0.3, np.float32)
    confidence = np.full_like(pseudo, 0.2)
    target, weight = combine_gold_and_pseudo(df, pseudo, confidence, gold_weight=8.0)
    assert target[0, 0] == 1.0 and weight[0, 0] == 8.0
    assert np.allclose(target[0, 1:], 0.3) and np.allclose(weight[0, 1:], 0.2)


def test_macro_weighted_bce_gives_targets_equal_importance():
    # Target 0 has much more supervision mass than target 1. Per-target
    # normalization means this does not make target 0 dominate the mean.
    logits = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    weights = torch.tensor([[100.0, 1.0], [100.0, 1.0]])
    loss = macro_weighted_bce(logits, target, weights)
    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-6)


def test_macro_weighted_bce_ignores_targets_with_no_supervision():
    logits = torch.zeros(2, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = macro_weighted_bce(logits, target, weights)
    assert torch.isfinite(loss)
    loss.backward()


def test_ranking_loss_ignores_low_confidence_pseudo_cells():
    logits = torch.tensor([[0.0], [1.0]], requires_grad=True)
    target = torch.tensor([[0.1], [0.9]])
    weight = torch.tensor([[0.1], [0.1]])
    loss = confidence_gated_ranking_loss(logits, target, weight, min_confidence=0.35)
    assert loss.item() == 0.0


def test_ranking_loss_uses_trusted_pairs():
    logits = torch.tensor([[0.0], [1.0]], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]])
    weight = torch.tensor([[8.0], [8.0]])
    loss = confidence_gated_ranking_loss(logits, target, weight, min_confidence=0.35)
    assert torch.isfinite(loss) and loss.item() > 0.0
