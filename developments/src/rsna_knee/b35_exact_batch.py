"""Numerically matched B35 encoder pass.

The first B35 implementation flattened all 32 sampled centres before splitting
ConvNeXt work into encoder batches.  The frozen B34 reference flattens only its
historical 16 centres.  Under BF16/cuDNN, changing those chunk boundaries can
change rounding slightly even though the images and weights are identical.

This subclass keeps the one-pass-per-image B35 design but encodes the historical
16 centres and the 16 extra centres as two groups.  The historical group now has
exactly the same flatten order and encoder chunk boundaries as ordinary B34, so
the zero-gated base equivalence check tests the model rather than a batch-shape
numerical artefact.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .b35_target_spatial_residual import (
    B35_BASE_SLICES,
    B35_DENSE_SLICES,
    B35TargetSpatialResidual,
)


class B35TargetSpatialResidualExactBatch(B35TargetSpatialResidual):
    """B35 with B34-matched chunking for the first 16 centres."""

    @torch.no_grad()
    def _encode_active_group(
        self,
        active_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode [N,G,3,H,W] while preserving per-series centre order."""
        if active_group.ndim != 5:
            raise ValueError("B35 active group must be [N,G,3,H,W]")
        n, g, c, h, w = active_group.shape
        if c != 3:
            raise ValueError("B35 active group must contain 3-channel triplets")

        encoder = self.base.encoder
        d = int(encoder.out_dim)
        r = int(self.head.n_regions)
        flat = active_group.reshape(n * g, c, h, w)
        global_blocks: list[torch.Tensor] = []
        spatial_blocks: list[torch.Tensor] = []

        # This split is deliberately identical to B34's _encode_slices path for
        # the historical 16-centre group: same flatten order, same chunk size.
        for chunk in flat.split(int(self.base.encoder_batch_size), dim=0):
            normalized = encoder._normalize(chunk)
            fmap = encoder.features(normalized)
            global_feature = encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
                chunk.shape[0], d
            )
            pooled = F.adaptive_avg_pool2d(
                fmap,
                (int(self.head.grid_size), int(self.head.grid_size)),
            )
            normalized_grid = encoder.pre_classifier[0](pooled)
            spatial_feature = normalized_grid.permute(0, 2, 3, 1).reshape(
                chunk.shape[0], r, d
            )
            global_blocks.append(global_feature)
            spatial_blocks.append(spatial_feature)

        global_group = torch.cat(global_blocks, dim=0).reshape(n, g, d)
        spatial_group = torch.cat(spatial_blocks, dim=0).reshape(n, g, r, d)
        return global_group, spatial_group

    @torch.no_grad()
    def _encode_combined(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if volumes.ndim != 6:
            raise ValueError("B35 expects [B,K,32,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if s != B35_DENSE_SLICES or c != 3:
            raise ValueError("B35 requires exactly 32 sampled 3-channel positions")

        active_indices = torch.nonzero(
            present.reshape(-1) > 0,
            as_tuple=False,
        ).flatten()
        if active_indices.numel() == 0:
            raise RuntimeError("B35 batch has no readable MRI series")

        flat_series = volumes.reshape(b * k, s, c, h, w)
        active = flat_series.index_select(0, active_indices)

        # Crucial numerical contract: encode the historical group independently.
        # This reproduces ordinary B34's N_active*16 flattening and split points.
        base_global, base_spatial = self._encode_active_group(
            active[:, :B35_BASE_SLICES]
        )
        extra_global, extra_spatial = self._encode_active_group(
            active[:, B35_BASE_SLICES:]
        )
        global_active = torch.cat((base_global, extra_global), dim=1)
        spatial_active = torch.cat((base_spatial, extra_spatial), dim=1)

        d = int(self.base.encoder.out_dim)
        r = int(self.head.n_regions)
        all_global = global_active.new_zeros((b * k, s, d)).index_copy(
            0,
            active_indices,
            global_active,
        )
        all_spatial = spatial_active.new_zeros((b * k, s, r, d)).index_copy(
            0,
            active_indices,
            spatial_active,
        )
        return (
            all_global.reshape(b, k, s, d),
            all_spatial.reshape(b, k, s, r, d),
        )
