import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.constants import TARGETS
from rsna_knee.dicom import preprocess_triplets
from rsna_knee.model import KneeMILNet
from rsna_knee.report_labels import combine_gold_and_pseudo
from rsna_knee.training import confidence_gated_ranking_loss


def test_preprocess_triplets_shape_and_range():
    volume = np.arange(9 * 12 * 10, dtype=np.float32).reshape(9, 12, 10)
    out = preprocess_triplets(volume, n_slices=5, image_size=16, gap=1)
    assert out.shape == (5, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_hierarchical_model_forward_is_finite():
    model = KneeMILNet(
        n_streams=6,
        n_slices=4,
        pretrained_weights=False,
        normalize_input=False,
        encoder_batch_size=3,
        gradient_checkpointing=False,
    ).eval()
    x = torch.rand(2, 6, 4, 3, 64, 64)
    present = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0]], dtype=torch.float32
    )
    with torch.no_grad():
        logits = model(x, present)
    assert logits.shape == (2, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_model_allows_all_streams_missing_without_nan():
    model = KneeMILNet(
        n_streams=6,
        n_slices=2,
        pretrained_weights=False,
        normalize_input=False,
        encoder_batch_size=2,
        gradient_checkpointing=False,
    ).eval()
    x = torch.zeros(1, 6, 2, 3, 64, 64)
    present = torch.zeros(1, 6)
    with torch.no_grad():
        logits = model(x, present)
    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_model_rejects_wrong_slice_count():
    model = KneeMILNet(
        n_streams=6,
        n_slices=4,
        pretrained_weights=False,
        normalize_input=False,
        gradient_checkpointing=False,
    )
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

    assert target[0, 0] == 1.0
    assert weight[0, 0] == 8.0
    assert np.allclose(target[0, 1:], 0.3)
    assert np.allclose(weight[0, 1:], 0.2)


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
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
