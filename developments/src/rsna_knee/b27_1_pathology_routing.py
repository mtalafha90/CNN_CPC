"""B27.1 collinearity-safe pathology-specific acquisition routing.

B27 exposed plane, fluid-sensitivity and fat-suppression metadata as three
independent additive pathology-routing terms.  A pre-outcome audit of the exact
17,475-series B20 training surface showed perfect collinearity between the two
sequence flags:

    (fluid_id, fat_id) = (1,1) for 7,459 series
    (fluid_id, fat_id) = (2,2) for 10,016 series
    discordant pairs   = 0

Because B27 initialised the fluid and fat tables identically at zero and both
saw the same categories on every training series, the two tables received the
same gradients and remained identical.  Their sum therefore double-counted one
empirical sequence axis.

B27.1 fixes that *before any B27 expert/gold evaluation* by representing the
paired sequence state once:

    12 targets x (3 plane + 2 paired-sequence categories) = 60 parameters.

The paired sequence category is defined only when the two source flags agree:
1 = structural + not-fat-suppressed, 2 = fluid-sensitive + fat-suppressed.
Unknown or discordant flag pairs receive a permanently fixed zero routing bias,
which makes inference conservative if a future/test surface differs from the
training metadata geometry.

All routing parameters are zero-initialised.  With shared B20 state and zero
routing, B27.1 is functionally identical to B20.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .b12_1_hierarchical import HierarchicalSeriesKneeMILNet, b12_1_model_spec
from .constants import N_TARGETS, TARGETS

B27_1_ARCHITECTURE = "hierarchical_series_token_pathology_paired_metadata_routing_v1"
B27_1_AGGREGATION = "b12_1_cross_attention_plus_zero_init_plane_paired_sequence_bias_v1"
B27_1_ROUTING_VERSION = "pathology_plane_paired_sequence_attention_bias_v1"
B27_1_ROUTE_PARAMETER_COUNT = N_TARGETS * (3 + 2)

PLANE_NAMES = ("Sagittal", "Coronal", "Axial")
PAIRED_SEQUENCE_NAMES = (
    "structural_non_fat_suppressed",
    "fluid_sensitive_fat_suppressed",
)


class PathologyPairedMetadataRoutedKneeMILNet(HierarchicalSeriesKneeMILNet):
    """B20 hierarchy plus plane and one non-redundant paired-sequence route."""

    def __init__(self, *args, **kwargs) -> None:
        # Parent first: identical RNG order for every B20-shared parameter.
        super().__init__(*args, **kwargs)
        self.route_plane_bias = nn.Parameter(torch.zeros(N_TARGETS, 3))
        self.route_sequence_bias = nn.Parameter(torch.zeros(N_TARGETS, 2))

    @staticmethod
    def _with_unknown_zero(table: torch.Tensor) -> torch.Tensor:
        zero = table.new_zeros((table.shape[0], 1))
        return torch.cat((zero, table), dim=1)

    @staticmethod
    def _gather_target_bias(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if table.ndim != 2 or ids.ndim != 2:
            raise ValueError("routing table/ids rank mismatch")
        b, k = ids.shape
        gathered = table[:, ids.reshape(-1)]
        return gathered.reshape(table.shape[0], b, k).permute(1, 0, 2)

    @staticmethod
    def paired_sequence_ids(series_meta: torch.Tensor) -> torch.Tensor:
        """Map [plane, fluid, fat] metadata to one conservative paired axis.

        Returns 1 or 2 only when fluid and fat flags agree on a known category.
        Unknown or discordant pairs map to 0, whose bias is fixed at zero.
        """
        if series_meta.ndim != 3 or series_meta.shape[-1] != 3:
            raise ValueError("B27.1 series_meta must have shape [B,K,3]")
        fluid = series_meta[:, :, 1].long()
        fat = series_meta[:, :, 2].long()
        valid = fluid.eq(fat) & (fluid.eq(1) | fluid.eq(2))
        return torch.where(valid, fluid, torch.zeros_like(fluid))

    def metadata_route_bias(self, series_meta: torch.Tensor) -> torch.Tensor:
        if series_meta.ndim != 3 or series_meta.shape[-1] != 3:
            raise ValueError("B27.1 series_meta must have shape [B,K,3]")
        plane_ids = series_meta[:, :, 0].long().clamp(0, 3)
        sequence_ids = self.paired_sequence_ids(series_meta)
        plane = self._gather_target_bias(
            self._with_unknown_zero(self.route_plane_bias), plane_ids
        )
        sequence = self._gather_target_bias(
            self._with_unknown_zero(self.route_sequence_bias), sequence_ids
        )
        return plane + sequence

    def _cross_attention_mask(
        self,
        series_meta: torch.Tensor,
        padding: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bias = self.metadata_route_bias(series_meta).to(dtype=dtype)
        if padding.shape != (bias.shape[0], bias.shape[2]):
            raise ValueError("B27.1 padding shape does not match routing surface")
        bias = bias.masked_fill(padding[:, None, :], float("-inf"))
        heads = int(self.cross_attention.num_heads)
        b, t, k = bias.shape
        return bias[:, None, :, :].expand(b, heads, t, k).reshape(b * heads, t, k)

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
        def table(param: torch.Tensor, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
            values = param.detach().float().cpu()
            return {
                target: {name: float(values[i, j]) for j, name in enumerate(names)}
                for i, target in enumerate(TARGETS)
            }

        return {
            "version": B27_1_ROUTING_VERSION,
            "parameter_count": B27_1_ROUTE_PARAMETER_COUNT,
            "unknown_or_discordant_sequence_bias": 0.0,
            "plane": table(self.route_plane_bias, PLANE_NAMES),
            "paired_sequence": table(self.route_sequence_bias, PAIRED_SEQUENCE_NAMES),
        }


def b27_1_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b12_1_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B27_1_ARCHITECTURE
    spec["aggregation"] = B27_1_AGGREGATION
    spec["b27_1_routing_version"] = B27_1_ROUTING_VERSION
    spec["b27_1_route_parameter_count"] = B27_1_ROUTE_PARAMETER_COUNT
    spec["b27_1_unknown_or_discordant_sequence_bias"] = 0.0
    return spec


def build_b27_1_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> PathologyPairedMetadataRoutedKneeMILNet:
    if spec.get("architecture") != B27_1_ARCHITECTURE:
        raise ValueError("not a B27.1 paired-metadata routing model spec")
    if spec.get("aggregation") != B27_1_AGGREGATION:
        raise ValueError("B27.1 aggregation policy mismatch")
    if spec.get("b27_1_routing_version") != B27_1_ROUTING_VERSION:
        raise ValueError("B27.1 routing version mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = PathologyPairedMetadataRoutedKneeMILNet(
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
