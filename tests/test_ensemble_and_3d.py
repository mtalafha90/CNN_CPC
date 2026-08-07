import numpy as np
import pandas as pd
import torch

from rsna_knee.constants import TARGETS
from rsna_knee.ensemble import ensemble_predictions
from rsna_knee.model import MultiSeriesKneeNet


def test_small_3d_arm_forward_shape():
    model = MultiSeriesKneeNet(
        n_streams=3,
        backbone="3d",
        in_channels=1,
        target_attention=True,
    )
    x = torch.rand(2, 3, 8, 64, 64)
    present = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x, present)
    assert out.shape == (2, len(TARGETS))
    assert torch.isfinite(out).all()


def _prediction_frame(offset=0.0):
    df = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"]})
    base = np.array([0.1, 0.5, 0.9]) + offset
    for j, target in enumerate(TARGETS):
        df[target] = np.clip(base + j * 1e-4, 0, 1)
    return df


def test_rank_ensemble_preserves_ids_and_range(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _prediction_frame(0.0).to_csv(a, index=False)
    _prediction_frame(-0.02).iloc[::-1].to_csv(b, index=False)
    out = ensemble_predictions([a, b], method="rank")
    assert out["StudyInstanceUID"].tolist() == ["a", "b", "c"]
    values = out[TARGETS].to_numpy(float)
    assert np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()
    assert values[0, 0] < values[1, 0] < values[2, 0]
