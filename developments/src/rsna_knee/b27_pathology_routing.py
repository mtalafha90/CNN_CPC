"""B27 pathology-specific acquisition routing on the frozen B20 family.

B20/B12.1 already has pathology query tokens that cross-attend to contextualised
series tokens.  B27 therefore does *not* add a second large routing network.
Instead it makes one much narrower change: each pathology may learn an additive
attention-logit bias from the observed acquisition metadata of each series
(plane, fluid sensitivity and fat suppression).

The routing tables are zero-initialised.  Consequently a freshly constructed
B27 model is functionally identical to B20 when the shared state is identical;
training can only move away from B20 by learning the 84 routing parameters:

    12 targets x (3 plane + 2 fluid + 2 fat categories) = 84 parameters.

Unknown metadata and padding always receive exactly zero routing bias.  No
anatomical preference is hard-coded and no LLM output enters the model.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .b12_1_hierarchical import (
    HierarchicalSeriesKneeMILNet,
    b12_1_model_spec,
)
from .constants import N_TARGETS, TARGETS

B27_ARCHITECTURE = "hierarchical_series_token_pathology_metadata_routing_v1"
B27_AGGREGATION = "b12_1_cross_attention_plus_zero_init_metadata_logit_bias_v1"
B27_ROUTING_VERSION = "pathology_metadata_attention_bias_v1"
B27_ROUTE_PARAMETER_COUNT = N_TARGETS * (3 + 2 + 2)

PLANE_NAMES = ("Sagittal", "Coronal", "Axial")
FLUID_NAMES = ("structural", "fluid_sensitive")
FAT_NAMES = ("not_fat_suppressed", "fat_suppressed")


class PathologyMetadataRoutedKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus a tiny pathology-specific metadata attention bias.

    The parent class still performs:

        slices -> learned series token -> shared study Transformer
        -> pathology queries -> multi-head cross attention -> 12 logits

    B27 only adds a per-pathology bias to the final cross-attention logits.  The
    bias is the sum of a plane term, a fluid-sensitivity term and a fat-
    suppression term for each real series.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Construct all B20/B12.1 parameters first, in exactly the same order.
        # With the same RNG seed, every shared parameter therefore receives the
        # same initial value as B20 before the new B27 parameters are created.
        super().__init__(*args, **kwargs)

        # Parameters exclude category 0.  Unknown/padding is represented by a
        # non-parameter zero column assembled at runtime, so it can never learn
        # a site/missingness shortcut.
        self.route_plane_bias = nn.Parameter(torch.zeros(N_TARGETS, 3))
        self.route_fluid_bias = nn.Parameter(torch.zeros(N_TARGETS, 2))
        self.route_fat_bias = nn.Parameter(torch.zeros(N_TARGETS, 2))

    @staticmethod
    def _with_unknown_zero(table: torch.Tensor) -> torch.Tensor:
        zero = table.new_zeros((table.shape[0], 1))
        return torch.cat((zero, table), dim=1)

    @staticmethod
    def _gather_target_bias(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Gather a [T,C] table with [B,K] ids -> [B,T,K]."""
        if table.ndim != 2 or ids.ndim != 2:
            raise ValueError("routing table/ids rank mismatch")
        b, k = ids.shape
        gathered = table[:, ids.reshape(-1)]
        return gathered.reshape(table.shape[0], b, k).permute(1, 0, 2)

    def metadata_route_bias(self, series_meta: torch.Tensor) -> torch.Tensor:
        """Return learned additive routing logits with shape [B,12,K]."""
        if series_meta.ndim != 3 or series_meta.shape[-1] != 3:
            raise ValueError("B27 series_meta must have shape [B,K,3]")

        plane_ids = series_meta[:, :, 0].long().clamp(0, 3)
        fluid_ids = series_meta[:, :, 1].long().clamp(0, 2)
        fat_ids = series_meta[:, :, 2].long().clamp(0, 2)

        plane = self._gather_target_bias(
            self._with_unknown_zero(self.route_plane_bias), plane_ids
        )
        fluid = self._gather_target_bias(
            self._with_unknown_zero(self.route_fluid_bias), fluid_ids
        )
        fat = self._gather_target_bias(
            self._with_unknown_zero(self.route_fat_bias), fat_ids
        )
        return plane + fluid + fat

    def _cross_attention_mask(
        self,
        series_meta: torch.Tensor,
        padding: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the per-batch/per-head additive mask expected by MHA."""
        bias = self.metadata_route_bias(series_meta).to(dtype=dtype)
        if padding.shape != (bias.shape[0], bias.shape[2]):
            raise ValueError("B27 padding shape does not match routing surface")

        # _study_memory returns a safe padding mask: even a genuinely empty
        # study has one synthetic zero token left unmasked, avoiding all -inf.
        bias = bias.masked_fill(padding[:, None, :], float("-inf"))
        heads = int(self.cross_attention.num_heads)
        b, t, k = bias.shape
        return (
            bias[:, None, :, :]
            .expand(b, heads, t, k)
            .reshape(b * heads, t, k)
        )

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        memory, padding, empty = self._study_memory(volumes, present, series_meta)
        b = memory.shape[0]
        queries = self.pathology_tokens[None, :, :].expand(b, -1, -1)
        queries = self.pathology_context(queries)

        route_mask = self._cross_attention_mask(
            series_meta,
            padding,
            dtype=queries.dtype,
        )
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            attn_mask=route_mask,
            need_weights=False,
        )
        queries = self.dropout(self.query_norm(queries + attended))
        logits = (queries * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        return torch.where(empty[:, None], self.target_bias[None, :], logits)

    def routing_tables(self) -> dict:
        """Return the learned 84 biases as a JSON-serialisable diagnostic."""
        def table(param: torch.Tensor, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
            values = param.detach().float().cpu()
            return {
                target: {name: float(values[i, j]) for j, name in enumerate(names)}
                for i, target in enumerate(TARGETS)
            }

        return {
            "version": B27_ROUTING_VERSION,
            "parameter_count": B27_ROUTE_PARAMETER_COUNT,
            "unknown_metadata_bias": 0.0,
            "plane": table(self.route_plane_bias, PLANE_NAMES),
            "fluid": table(self.route_fluid_bias, FLUID_NAMES),
            "fat": table(self.route_fat_bias, FAT_NAMES),
        }


def b27_model_spec(config: dict, *, normalize_input: bool) -> dict:
    """B12.1/B20 model spec with only the routing identity changed."""
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B27_ARCHITECTURE
    spec["aggregation"] = B27_AGGREGATION
    spec["b27_routing_version"] = B27_ROUTING_VERSION
    spec["b27_route_parameter_count"] = B27_ROUTE_PARAMETER_COUNT
    spec["b27_routing_unknown_metadata_bias"] = 0.0
    return spec


def build_b27_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> PathologyMetadataRoutedKneeMILNet:
    if spec.get("architecture") != B27_ARCHITECTURE:
        raise ValueError("not a B27 pathology-routing model spec")
    if spec.get("aggregation") != B27_AGGREGATION:
        raise ValueError("B27 aggregation policy mismatch")
    if spec.get("b27_routing_version") != B27_ROUTING_VERSION:
        raise ValueError("B27 routing version mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = PathologyMetadataRoutedKneeMILNet(
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=bool(pretrained_weights),
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
        series_pool_heads=int(spec["series_pool_heads"]),
    )
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
