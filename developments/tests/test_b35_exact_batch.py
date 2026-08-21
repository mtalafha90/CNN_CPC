from __future__ import annotations

import torch
from torch import nn

from rsna_knee.b35_exact_batch import B35TargetSpatialResidualExactBatch
from rsna_knee.b35_target_spatial_residual import B35_BASE_SLICES, B35_DENSE_SLICES


class _ChannelNorm2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Mirror ConvNeXt's channel-wise LayerNorm on NCHW tensors.
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _DummyEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.out_dim = dim
        self.features = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_classifier = nn.Sequential(_ChannelNorm2d(dim), nn.Flatten(1))

    def _normalize(self, x):
        return x

    def forward(self, x):
        return self.pre_classifier(self.avgpool(self.features(x)))


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyEncoder()
        self.encoder_batch_size = 3
        self.n_slices = B35_BASE_SLICES


def test_exact_batch_first_16_match_direct_encoder_path():
    torch.manual_seed(7)
    base = _DummyBase()
    model = B35TargetSpatialResidualExactBatch(base)

    # Two active series force multiple encoder chunks and a non-full final chunk.
    volumes = torch.randn(1, 2, B35_DENSE_SLICES, 3, 16, 16)
    present = torch.ones(1, 2)

    global_feature, spatial = model._encode_combined(volumes, present)

    historical = volumes[:, :, :B35_BASE_SLICES].reshape(-1, 3, 16, 16)
    direct = torch.cat(
        [base.encoder(chunk) for chunk in historical.split(base.encoder_batch_size)],
        dim=0,
    ).reshape(1, 2, B35_BASE_SLICES, base.encoder.out_dim)

    torch.testing.assert_close(
        global_feature[:, :, :B35_BASE_SLICES],
        direct,
        rtol=0,
        atol=0,
    )
    assert global_feature.shape == (1, 2, B35_DENSE_SLICES, base.encoder.out_dim)
    assert spatial.shape[:4] == (1, 2, B35_DENSE_SLICES, 9)
