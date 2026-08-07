import numpy as np
import pandas as pd
import torch

from rsna_knee.constants import TARGETS
from rsna_knee.dicom import preprocess_triplets
from rsna_knee.model import MultiSeriesKneeNet, TopKAttentionPool
from rsna_knee.report_labels import combine_gold_and_pseudo


def test_preprocess_triplets_shape_and_range():
    volume = np.arange(9 * 12 * 10, dtype=np.float32).reshape(9, 12, 10)
    out = preprocess_triplets(volume, n_slices=5, image_size=16, gap=1)
    assert out.shape == (5, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_2p5d_target_attention_forward_shape():
    model = MultiSeriesKneeNet(
        n_streams=3,
        pretrained=False,
        in_channels=3,
        target_attention=True,
    )
    x = torch.rand(2, 3, 4, 3, 64, 64)
    present = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x, present)
    assert logits.shape == (2, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_topk_pool_returns_fixed_feature_shape():
    pool = TopKAttentionPool(dim=32, fraction=0.25)
    x = torch.randn(3, 12, 32)
    out = pool(x)
    assert out.shape == (3, 32)


def test_partial_gold_overrides_only_annotated_cells():
    df = pd.DataFrame({"StudyInstanceUID": ["x"], "Report": [""]})
    for t in TARGETS:
        df[t] = np.nan
    df.loc[0, "ACL"] = 1.0

    pseudo = np.full((1, len(TARGETS)), 0.3, np.float32)
    conf = np.full_like(pseudo, 0.2)
    target, weight = combine_gold_and_pseudo(df, pseudo, conf, gold_weight=8.0)

    assert target[0, 0] == 1.0
    assert weight[0, 0] == 8.0
    assert np.allclose(target[0, 1:], 0.3)
    assert np.allclose(weight[0, 1:], 0.2)
