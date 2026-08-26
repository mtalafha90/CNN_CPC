"""Generate the standalone B48-shaped Google Colab subset notebook.

The generated notebook deliberately reuses the old notebook's Drive archive
contract and DICOM preprocessing, but it is a separate artifact.  Its model is
a compact, freshly initialized teaching/sandbox implementation of B48's
matched global-query-conditioned sparse-MIL comparison; it is not the full
scanner-domain, weak-label B48 protocol in ``developments/``.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path


BASE_BUILDER = Path(__file__).with_name("build_notebook.py")
BASE_NAMESPACE = runpy.run_path(str(BASE_BUILDER))
CELLS: list[tuple[str, str]] = list(BASE_NAMESPACE["CELLS"])


def replace_cell(index: int, kind: str, text: str) -> None:
    """Replace one inherited notebook cell without modifying the old builder."""
    CELLS[index] = (kind, text.strip("\n"))


def replace_text(index: int, old: str, new: str) -> None:
    """Apply one checked text substitution to an inherited cell."""
    kind, text = CELLS[index]
    if old not in text:
        raise RuntimeError(f"Expected text was not found in inherited cell {index}")
    CELLS[index] = (kind, text.replace(old, new))


def build(path: Path) -> Path:
    """Write the B48 subset notebook JSON to ``path`` and return it."""
    cells = []
    for kind, text in CELLS:
        source = text.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return path


# The original notebook remains untouched.  Only this new notebook's copied
# cells are retitled/reconfigured below.
replace_cell(
    0,
    "markdown",
    '<a href="https://colab.research.google.com/github/mtalafha90/CNN_CPC/blob/main/notebook/b48_global_conditioned_sparse_mil_colab.ipynb" '
    'target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" '
    'alt="Open In Colab"/></a>',
)

replace_cell(
    1,
    "markdown",
    r"""
# Standalone B48-shaped global-query-conditioned sparse-MIL knee MRI sandbox

This is a separate, self-contained Google Colab notebook for the same small
MRI subset already stored on Google Drive. It preserves the earlier notebook's
bounded DICOM pipeline and native-aspect 448×448 geometry, then replaces the
old global-plus-local model with a matched B48-shaped comparison:

```text
same encoder and same train/validation split
      ├── static_prior_control
      │     pathology query before study-series cross-attention
      └── post_cross_attention_candidate
            pathology query after study-series cross-attention

both → detached 96-dimensional query/token cosine compatibility
     → target-specific 6×6 top-k local evidence
     → zero-start residual fusion with global logits
```

The notebook trains both arms from byte-identical initialization, uses the same
fixed split and data order for each arm, and saves their paired subset result.
It starts with a no-update preflight that checks the zero-start gates,
detachment boundary, and memory footprint before either optimizer takes a step.
""",
)

replace_cell(
    2,
    "markdown",
    r"""
## What this notebook does—and does not—mean

This notebook reflects the **B48 mechanism**: a pathology-specific global
query softly re-ranks local spatial tokens through

\[
e_t(x_i) + \tanh(a_t)\,\cos\!\left(W_q\operatorname{LN}(q_t),
W_k\operatorname{LN}(x_i)\right).
\]

`a_t` starts at zero for every target. Therefore each arm begins as ordinary
sparse MIL; the global-context term can influence token ranking only after the
gate learns to open. The global query is detached before it reaches the local
head, so the local auxiliary loss cannot train the global query branch through
that route.

This is a **compact subset sandbox**, not the official B48 result or a
replacement for its protocol. It creates fresh compact weights from the subset
CSV labels, has no Phase-9/B34 checkpoint, no report-only fill artifact, no
official-gold exclusion audit, and no frozen scanner-domain split. Do not use
its hold-out AUC to claim a B48 endpoint, choose a scientific winner, or change
the already prepared full B48 design. Its purpose is to verify the data path,
resource behavior, paired-arm logic, and global-to-spatial representation on
the same Drive subset before the actual B48 protocol is run.
""",
)

replace_cell(
    3,
    "markdown",
    r"""
## 1. Google Drive layout

Keep the same two supplied archives at the top level of Google Drive. This new
notebook mounts Drive, copies them to its own local Colab folder, safely unzips
them, and writes only to a new B48-specific output folder. It never edits the
earlier notebook or its `knee_mri_subset_outputs/` results.

```text
MyDrive/
├── colab_subset.zip                         # same labelled training subset
├── test.zip                                 # same unlabelled test subset
├── knee_mri_subset_outputs/                 # earlier notebook; untouched
└── knee_mri_b48_subset_outputs/             # created only by this notebook
```

Expected extracted training layout: `train.csv`, `train_series.csv`, and either
`train_series/` or `train_images/`. Expected extracted test layout: `test.csv`,
`test_series.csv`, and either `test_series/` or `test_images/`.

The train table must contain `StudyInstanceUID` plus the 12 binary target
columns. A target cell can be `0`, `1`, or blank; blank cells are excluded from
the masked weighted BCE loss. The test subset is used only after both arms
finish to produce one prediction CSV per arm.
""",
)

replace_cell(
    8,
    "code",
    r'''
@dataclass(frozen=True)
class B48SubsetReference:
    """Record the immutable scope boundary for this separate Colab sandbox."""

    experiment: str = "B48-shaped matched global-conditioned sparse-MIL subset sandbox"
    model_status: str = "fresh compact subset weights; no Phase-9/B34 checkpoint"
    arms: tuple[str, str] = (
        "static_prior_control",
        "post_cross_attention_candidate",
    )
    context_dim: int = 96
    context_metric: str = "cosine_low_rank_query_token_compatibility"
    query_gradient: str = "detached_before_local_head"
    official_b48_protocol: str = (
        "not run here: report-only weak labels, gold exclusion, and scanner-domain split are absent"
    )
    interpretation: str = "mechanism and resource check only; not an official B48 gate"


B48_SUBSET_REFERENCE = B48SubsetReference()


def display_b48_subset_reference() -> pd.DataFrame:
    """Display the subset-sandbox boundary before data and model construction."""
    table = pd.DataFrame([asdict(B48_SUBSET_REFERENCE)])
    display(table)
    return table


B48_SUBSET_REFERENCE_TABLE = display_b48_subset_reference()
''',
)

# Keep the exact Drive archives but isolate local extraction and persistent outputs.
replace_text(10, 'local_root=Path("/content/knee_mri_subset"),', 'local_root=Path("/content/knee_mri_b48_subset"),')
replace_text(10, 'DRIVE_ROOT / "MyDrive" / "knee_mri_subset_outputs"', 'DRIVE_ROOT / "MyDrive" / "knee_mri_b48_subset_outputs"')
replace_text(
    10,
    '    feature_dim: int = 128\n',
    '''    feature_dim: int = 128
    # Keep B48's fixed low-rank query/token compatibility dimension.
    context_dim: int = 96
    # Use one compact global series-memory transformer layer in this subset sandbox.
    global_memory_layers: int = 1
    # Split the 128-dimensional global representation into four attention heads.
    global_attention_heads: int = 4
    # Apply modest, paired dropout inside the compact global-memory branch.
    global_dropout: float = 0.10
''',
)
replace_text(
    10,
    "    # Run B37's fixed two-epoch duration by default; this is not a B37 reproduction.\n",
    "    # Run a fixed two-epoch matched subset duration; this is not the full B48 protocol.\n",
)

replace_cell(
    17,
    "markdown",
    "## 8. B48-shaped model: compact global pathology queries and conditioned sparse evidence",
)

replace_cell(
    18,
    "code",
    r'''
class ConvNormAct(nn.Module):
    """A convolution, group normalization, and GELU activation block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 8))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    """A compact residual block that preserves feature-map resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SliceEncoder(nn.Module):
    """Encode one 448×448 triplet into global and 6×6 local features."""

    def __init__(self, feature_dim: int, grid_size: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.grid_size = int(grid_size)
        self.stem = ConvNormAct(3, 32, stride=2)
        self.stage1 = nn.Sequential(ConvNormAct(32, 48, stride=2), ResidualBlock(48))
        self.stage2 = nn.Sequential(ConvNormAct(48, 72, stride=2), ResidualBlock(72))
        self.stage3 = nn.Sequential(ConvNormAct(72, 96, stride=2), ResidualBlock(96))
        self.stage4 = nn.Sequential(
            ConvNormAct(96, self.feature_dim, stride=2), ResidualBlock(self.feature_dim)
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(images)))))
        global_feature = F.adaptive_avg_pool2d(features, 1).flatten(1)
        local_feature = F.adaptive_avg_pool2d(features, self.grid_size)
        return global_feature, local_feature


def position_basis(position: torch.Tensor) -> torch.Tensor:
    """Create the continuous eight-dimensional slice-position representation."""
    z = position.float().clamp(0.0, 1.0)
    return torch.stack(
        [
            z,
            z.square(),
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2 * math.pi * z),
            torch.cos(2 * math.pi * z),
            torch.sin(4 * math.pi * z),
            torch.cos(4 * math.pi * z),
        ],
        dim=-1,
    )


@dataclass
class B48SubsetHeadOutput:
    """Sparse local logits plus optional context/top-k audit values."""

    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor
    context_abs_mean: torch.Tensor
    base_top_indices: torch.Tensor | None
    topk_overlap_with_static: torch.Tensor | None


@dataclass
class B48SubsetOutput:
    """Combined/global/local logits and the detached local-conditioning query."""

    logits: torch.Tensor
    global_logits: torch.Tensor
    local_logits: torch.Tensor
    context_query: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor
    context_abs_mean: torch.Tensor
    topk_overlap_with_static: torch.Tensor | None


class CompactGlobalPathologyBranch(nn.Module):
    """Compact B34-shaped series memory and pathology-query readout.

    This branch is newly initialized for the Drive subset.  It mirrors the B48
    *representation boundary*—static pathology prior versus a query after
    cross-attention over study-series memory—without claiming to be the frozen
    pretrained B34 hierarchy used by the full experiment.
    """

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        dim = int(config.feature_dim)
        heads = int(config.global_attention_heads)
        if dim % heads:
            raise ValueError("feature_dim must divide evenly across global_attention_heads")
        self.plane_embedding = nn.Embedding(4, dim, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, dim, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, dim, padding_idx=0)
        self.series_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=2 * dim,
            dropout=float(config.global_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=int(config.global_memory_layers))
        self.pathology_tokens = nn.Parameter(torch.empty(N_TARGETS, dim))
        self.pathology_context = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU()
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=float(config.global_dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(dim)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, dim))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.normal_(self.pathology_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.target_weight, mean=0.0, std=0.02)

    def forward(
        self,
        series_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prior queries, post-memory queries, and global logits."""
        if series_feature.ndim != 3:
            raise ValueError("global series features must be [B, S, D]")
        batch, series, _ = series_feature.shape
        if present.shape != (batch, series):
            raise ValueError("global present mask shape mismatch")
        metadata = (
            self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
            + self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
            + self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        ).to(series_feature.dtype)
        memory_input = self.series_norm(series_feature + metadata)
        memory_input = memory_input * present[:, :, None].to(memory_input.dtype)
        padding = present <= 0
        if bool(padding.all(dim=1).any()):
            raise RuntimeError("Each study needs at least one readable series")
        memory = self.context(memory_input, src_key_padding_mask=padding)
        memory = memory.masked_fill(padding[:, :, None], 0.0)
        raw_queries = self.pathology_tokens[None, :, :].expand(batch, -1, -1)
        prior = self.pathology_context(raw_queries)
        attended, _ = self.cross_attention(
            prior,
            memory,
            memory,
            key_padding_mask=padding,
            need_weights=False,
        )
        static_query = self.query_norm(prior)
        post_query = self.query_norm(prior + attended)
        global_logits = (
            post_query * self.target_weight[None, :, :]
        ).sum(dim=-1) + self.target_bias[None, :]
        return static_query, post_query, global_logits


class B48SubsetSparseEvidenceHead(nn.Module):
    """Top-k sparse MIL with B48's detached-query cosine residual."""

    def __init__(self, feature_dim: int, grid_size: int, top_k: int, context_dim: int = 96) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.grid_size = int(grid_size)
        self.top_k = int(top_k)
        self.context_dim = int(context_dim)
        self.n_regions = self.grid_size * self.grid_size
        if self.context_dim != 96:
            raise ValueError("This B48 subset notebook fixes context_dim=96")
        self.position_projection = nn.Linear(8, self.feature_dim, bias=False)
        self.region_embedding = nn.Parameter(torch.zeros(self.n_regions, self.feature_dim))
        self.plane_embedding = nn.Embedding(4, self.feature_dim, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, self.feature_dim, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, self.feature_dim, padding_idx=0)
        self.evidence_weight = nn.Parameter(torch.empty(N_TARGETS, self.feature_dim))
        self.evidence_bias = nn.Parameter(torch.zeros(N_TARGETS))
        self.context_query = nn.Linear(self.feature_dim, self.context_dim, bias=False)
        self.context_key = nn.Linear(self.feature_dim, self.context_dim, bias=False)
        self.context_gate = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.normal_(self.evidence_weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.context_query.weight)
        nn.init.xavier_uniform_(self.context_key.weight)

    def effective_context_gate(self) -> torch.Tensor:
        """Bound target-specific context contribution to [-1, 1]."""
        return torch.tanh(self.context_gate)

    def _tokens(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add local coordinate/metadata context and construct the valid-token mask."""
        batch, series, slices, regions, feature_dim = spatial.shape
        if regions != self.n_regions or feature_dim != self.feature_dim:
            raise ValueError("Sparse feature shape does not match the B48 head")
        tokens = F.layer_norm(spatial.float(), (feature_dim,)).to(spatial.dtype)
        position = self.position_projection(position_basis(slice_position)).to(tokens.dtype)
        metadata = (
            self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
            + self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
            + self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        ).to(tokens.dtype)
        tokens = tokens + position[:, :, :, None, :]
        tokens = tokens + metadata[:, :, None, None, :]
        tokens = tokens + self.region_embedding.to(tokens.dtype)[None, None, None, :, :]
        tokens = tokens.reshape(batch, series * slices * regions, feature_dim)
        invalid = (
            (present <= 0)[:, :, None, None]
            .expand(batch, series, slices, regions)
            .reshape(batch, series * slices * regions)
        )
        if int((~invalid).sum(dim=1).min().item()) < self.top_k:
            raise RuntimeError("There are fewer valid local-MIL tokens than top_k")
        return tokens, invalid

    def _context_residual(
        self,
        tokens: torch.Tensor,
        global_query: torch.Tensor,
        invalid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute B48's bounded detached-query compatibility residual."""
        batch, targets, feature_dim = global_query.shape
        if targets != N_TARGETS or feature_dim != self.feature_dim or batch != tokens.shape[0]:
            raise ValueError("B48 global-query shape does not match local tokens")
        # This is the explicit stop-gradient boundary required by B48.
        query = global_query.detach().float()
        query = F.layer_norm(query, (self.feature_dim,))
        token = F.layer_norm(tokens.float(), (self.feature_dim,))
        query = F.normalize(self.context_query(query), p=2.0, dim=-1, eps=1e-6)
        token = F.normalize(self.context_key(token), p=2.0, dim=-1, eps=1e-6)
        cosine = torch.einsum("btr,bnr->btn", query, token)
        residual = self.effective_context_gate().float()[None, :, None] * cosine
        valid = (~invalid).float()
        denominator = valid.sum(dim=-1).clamp_min(1.0)[:, None]
        context_abs_mean = (residual.abs() * valid[:, None, :]).sum(dim=-1) / denominator
        return residual, context_abs_mean

    def forward_details(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        global_query: torch.Tensor,
        *,
        audit_context: bool = False,
    ) -> B48SubsetHeadOutput:
        """Score all local tokens, rank top-k, and optionally audit rank changes."""
        tokens, invalid = self._tokens(spatial, present, series_meta, slice_position)
        base_score = torch.einsum(
            "bnd,td->btn", tokens, self.evidence_weight.to(tokens.dtype)
        ) + self.evidence_bias.to(tokens.dtype)[None, :, None]
        context_residual, context_abs_mean = self._context_residual(tokens, global_query, invalid)
        score = (base_score + context_residual.to(base_score.dtype)).masked_fill(
            invalid[:, None, :], float("-inf")
        )
        top_values, top_indices = torch.topk(score, k=self.top_k, dim=-1, largest=True, sorted=True)
        local_logits = torch.logsumexp(top_values.float(), dim=-1) - math.log(float(self.top_k))
        base_top_indices = overlap = None
        if audit_context:
            static_score = base_score.masked_fill(invalid[:, None, :], float("-inf"))
            base_top_indices = torch.topk(static_score, k=self.top_k, dim=-1, largest=True, sorted=True).indices
            overlap = (
                (top_indices[..., :, None] == base_top_indices[..., None, :])
                .any(dim=-1)
                .float()
                .mean(dim=-1)
            )
        return B48SubsetHeadOutput(
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values.float(),
            context_abs_mean=context_abs_mean.float(),
            base_top_indices=base_top_indices,
            topk_overlap_with_static=overlap,
        )


class B48SubsetModel(nn.Module):
    """Fresh compact model with the two B48 query-source arms."""

    ARMS = ("static_prior_control", "post_cross_attention_candidate")

    def __init__(self, config: RunConfig, arm: str) -> None:
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}; got {arm!r}")
        self.config = config
        self.arm = str(arm)
        self.context_source = (
            "pathology_prior_before_series_cross_attention"
            if arm == "static_prior_control"
            else "post_series_cross_attention_query"
        )
        self.encoder = SliceEncoder(config.feature_dim, config.grid_size)
        self.global_branch = CompactGlobalPathologyBranch(config)
        self.sparse_head = B48SubsetSparseEvidenceHead(
            config.feature_dim, config.grid_size, config.top_k, context_dim=config.context_dim
        )
        # As in B48, local logits enter the final prediction through a zero-start gate.
        self.fusion_gate = nn.Parameter(torch.zeros(N_TARGETS))

    def _encode_active_series(
        self, volumes: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode readable series only, then restore padded study shapes."""
        batch, series, slices, channels, height, width = volumes.shape
        if channels != 3:
            raise ValueError("The model expects three-channel 2.5D triplets")
        flat_series = volumes.reshape(batch * series, slices, channels, height, width)
        active_index = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        if active_index.numel() == 0:
            raise RuntimeError("The batch has no readable MRI series")
        active = flat_series.index_select(0, active_index)
        images = active.reshape(-1, channels, height, width)
        global_blocks, local_blocks = [], []
        for image_chunk in images.split(self.config.encoder_chunk_size, dim=0):
            if self.training and self.config.gradient_checkpointing:
                global_feature, local_feature = checkpoint(self.encoder, image_chunk, use_reentrant=False)
            else:
                global_feature, local_feature = self.encoder(image_chunk)
            global_blocks.append(global_feature)
            local_blocks.append(local_feature)
        global_active = torch.cat(global_blocks, dim=0).reshape(
            active.shape[0], slices, self.config.feature_dim
        )
        local_active = torch.cat(local_blocks, dim=0).reshape(
            active.shape[0], slices, self.config.feature_dim, self.config.grid_size, self.config.grid_size
        )
        global_all = global_active.new_zeros((batch * series, slices, self.config.feature_dim))
        global_all.index_copy_(0, active_index, global_active)
        local_all = local_active.new_zeros(
            (batch * series, slices, self.config.feature_dim, self.config.grid_size, self.config.grid_size)
        )
        local_all.index_copy_(0, active_index, local_active)
        global_feature = global_all.reshape(batch, series, slices, self.config.feature_dim)
        spatial_feature = local_all.reshape(
            batch, series, slices, self.config.feature_dim, self.config.grid_size, self.config.grid_size
        ).permute(0, 1, 2, 4, 5, 3).reshape(
            batch, series, slices, self.config.grid_size * self.config.grid_size, self.config.feature_dim
        )
        return global_feature, spatial_feature

    def forward(
        self,
        volumes: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        *,
        audit_context: bool = False,
    ) -> B48SubsetOutput:
        """Return B48-shaped combined/global/local study predictions."""
        global_feature, spatial_feature = self._encode_active_series(volumes, present)
        series_feature = global_feature.mean(dim=2) * present[:, :, None].to(global_feature.dtype)
        static_query, post_query, global_logits = self.global_branch(
            series_feature, present, series_meta
        )
        # Select only the query source under comparison, then detach it before local scoring.
        context_query = (
            static_query if self.arm == "static_prior_control" else post_query
        ).detach()
        details = self.sparse_head.forward_details(
            spatial_feature,
            present,
            series_meta,
            slice_position,
            context_query,
            audit_context=audit_context,
        )
        logits = global_logits.float() + torch.tanh(self.fusion_gate)[None, :] * details.local_logits.float()
        return B48SubsetOutput(
            logits=logits,
            global_logits=global_logits,
            local_logits=details.local_logits,
            context_query=context_query,
            top_indices=details.top_indices,
            top_values=details.top_values,
            context_abs_mean=details.context_abs_mean,
            topk_overlap_with_static=details.topk_overlap_with_static,
        )
''',
)

replace_cell(
    19,
    "markdown",
    "## 9. Paired-arm loss, preflight, training, and comparison functions",
)

replace_cell(
    20,
    "code",
    r'''
def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted BCE only for non-blank subset target cells."""
    known = torch.isfinite(target)
    if not bool(known.any()):
        raise RuntimeError("This batch has no usable supervision cells")
    safe_target = torch.nan_to_num(target, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), safe_target.float(), pos_weight=positive_weight.float(), reduction="none"
    )
    return (loss * known).sum() / known.sum().clamp_min(1)


def make_positive_weight(frame: pd.DataFrame) -> torch.Tensor:
    """Build clipped target-wise positive weights from the common train rows."""
    labels = frame[TARGETS].apply(pd.to_numeric, errors="coerce")
    known = labels.notna().sum(axis=0).to_numpy(np.float32)
    positive = labels.fillna(0).sum(axis=0).to_numpy(np.float32)
    negative = np.maximum(known - positive, 1.0)
    weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 20.0)
    return torch.tensor(weight, dtype=torch.float32, device=DEVICE)


def binary_auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    """Compute ROC-AUC without an additional dependency; return None for one class."""
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    positive = int(target.sum())
    negative = int(len(target) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(probability), dtype=np.float64)
    ranks[order] = np.arange(1, len(probability) + 1, dtype=np.float64)
    ordered_probability = probability[order]
    start = 0
    while start < len(ordered_probability):
        end = start + 1
        while end < len(ordered_probability) and ordered_probability[end] == ordered_probability[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    return float((ranks[target == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def evaluate_predictions(target: np.ndarray, probability: np.ndarray) -> dict:
    """Calculate defined per-target and macro AUC values for the subset split."""
    per_target: dict[str, float | None] = {}
    for index, name in enumerate(TARGETS):
        known = np.isfinite(target[:, index])
        per_target[name] = (
            binary_auc(target[known, index].astype(int), probability[known, index])
            if known.any() else None
        )
    defined = [value for value in per_target.values() if value is not None]
    return {
        "mean_auc": None if not defined else float(np.mean(defined)),
        "per_target_auc": per_target,
        "known_cells": int(np.isfinite(target).sum()),
    }


def move_model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move only B48 model inputs to GPU/CPU without pinned-memory pressure."""
    return (
        batch["volumes"].to(DEVICE, non_blocking=False),
        batch["present"].to(DEVICE, non_blocking=False),
        batch["series_meta"].to(DEVICE, non_blocking=False),
        batch["slice_position"].to(DEVICE, non_blocking=False),
    )


def move_batch(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move one labelled batch in the model-forward/loss order."""
    volumes, present, metadata, position = move_model_inputs(batch)
    return volumes, present, metadata, position, batch["target"].to(DEVICE, non_blocking=False)


def autocast_context():
    """Use CUDA fp16 autocast when available and ordinary precision otherwise."""
    return torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()


@dataclass
class B48SubsetExperiment:
    """All run objects for one arm of the matched Drive-subset comparison."""

    arm: str
    paths: DrivePaths
    config: RunConfig
    model: B48SubsetModel
    optimizer: torch.optim.Optimizer
    scaler: object
    train_loader: DataLoader
    validation_loader: DataLoader | None
    positive_weight: torch.Tensor
    train_uid_sha256: str
    validation_uid_sha256: str
    history: list[dict] = field(default_factory=list)


@dataclass
class B48MatchedPair:
    """The two B48 query-source arms sharing one split and initialization."""

    static_prior_control: B48SubsetExperiment
    post_cross_attention_candidate: B48SubsetExperiment
    initialization_fingerprint: str

    def arms(self) -> tuple[B48SubsetExperiment, B48SubsetExperiment]:
        return self.static_prior_control, self.post_cross_attention_candidate


def _uid_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint the exact study membership of a split without exposing data."""
    payload = "\n".join(frame["StudyInstanceUID"].astype(str).tolist()) + "\n"
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()


def _model_fingerprint(model: nn.Module) -> str:
    """Fingerprint all initialized tensors to prove the two arm states match."""
    digest = __import__("hashlib").sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _make_loader(dataset: Dataset, config: RunConfig, *, shuffle: bool, seed: int) -> DataLoader:
    """Build an arm-private loader whose shuffled study order is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=collate_studies,
    )


def _construct_matched_model(config: RunConfig, arm: str) -> B48SubsetModel:
    """Construct either arm under the same private RNG stream."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed) + 481516)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed) + 481516)
        model = B48SubsetModel(config, arm)
    return model.to(DEVICE)


def _prepare_subset_split(paths: DrivePaths, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict]]]:
    """Read the shared subset once and make the common deterministic split."""
    validate_dataset(paths)
    train_table = pd.read_csv(paths.train_csv)
    series_table = pd.read_csv(paths.series_csv)
    train_table["StudyInstanceUID"] = train_table["StudyInstanceUID"].astype(str)
    records = build_series_records(series_table, config)
    labels = train_table[TARGETS].apply(pd.to_numeric, errors="coerce")
    usable = train_table["StudyInstanceUID"].isin(records) & labels.notna().any(axis=1)
    usable_table = train_table.loc[usable].reset_index(drop=True)
    if usable_table.empty:
        raise ValueError("No studies remain after matching labels to readable MRI metadata")
    train_frame, validation_frame = make_split(
        usable_table, config.validation_fraction, config.seed
    )
    return train_frame, validation_frame, records


def build_b48_matched_pair(paths: DrivePaths, config: RunConfig = CONFIG) -> B48MatchedPair:
    """Build both arms from exactly the same subset split and initial state."""
    set_seed(config.seed)
    train_frame, validation_frame, records = _prepare_subset_split(paths, config)
    train_dataset = KneeMRIDataset(
        train_frame, records, paths, config, split="train", include_targets=True
    )
    validation_dataset = (
        KneeMRIDataset(validation_frame, records, paths, config, split="train", include_targets=True)
        if not validation_frame.empty else None
    )
    positive_weight = make_positive_weight(train_frame)
    train_uid_sha256, validation_uid_sha256 = _uid_sha256(train_frame), _uid_sha256(validation_frame)

    def make_arm(arm: str) -> B48SubsetExperiment:
        model = _construct_matched_model(config, arm)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        return B48SubsetExperiment(
            arm=arm,
            paths=paths,
            config=config,
            model=model,
            optimizer=optimizer,
            scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda"),
            train_loader=_make_loader(train_dataset, config, shuffle=True, seed=config.seed + 91),
            validation_loader=(
                _make_loader(validation_dataset, config, shuffle=False, seed=config.seed + 92)
                if validation_dataset is not None else None
            ),
            positive_weight=positive_weight,
            train_uid_sha256=train_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
        )

    control = make_arm("static_prior_control")
    candidate = make_arm("post_cross_attention_candidate")
    control_fingerprint = _model_fingerprint(control.model)
    candidate_fingerprint = _model_fingerprint(candidate.model)
    if control_fingerprint != candidate_fingerprint:
        raise RuntimeError("Matched B48 arms did not start from identical parameter tensors")
    print(
        f"train studies={len(train_dataset)} | validation studies={0 if validation_dataset is None else len(validation_dataset)} | "
        f"matched_initialization={control_fingerprint[:16]}"
    )
    return B48MatchedPair(control, candidate, control_fingerprint)


def _has_nonzero_gradient(parameters: Iterable[nn.Parameter]) -> bool:
    """Return whether any supplied parameter has a nonzero gradient."""
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )


def check_zero_start_pair_equivalence(pair: B48MatchedPair) -> float:
    """Verify that the two arms are numerically identical while both gates are closed.

    The probe is synthetic and small, so it does not consume a shuffled DICOM
    batch or disturb either arm's future train-loader order.  At initialization
    the models have identical tensors, the local context gate is zero, and the
    final local-fusion gate is zero.  Therefore static and post-attention query
    selection must have no effect on any output logit yet.
    """
    control, candidate = pair.arms()
    was_training = (control.model.training, candidate.model.training)
    control.model.eval()
    candidate.model.eval()
    probe_side, probe_slices = 192, 2
    volumes = torch.zeros((1, 1, probe_slices, 3, probe_side, probe_side), device=DEVICE)
    present = torch.ones((1, 1), dtype=torch.float32, device=DEVICE)
    metadata = torch.zeros((1, 1, 3), dtype=torch.long, device=DEVICE)
    position = torch.linspace(0.0, 1.0, probe_slices, device=DEVICE)[None, None, :]
    with torch.no_grad(), autocast_context():
        control_output = control.model(volumes, present, metadata, position)
        candidate_output = candidate.model(volumes, present, metadata, position)
    max_abs = max(
        float((control_output.logits.float() - candidate_output.logits.float()).abs().max().cpu()),
        float((control_output.global_logits.float() - candidate_output.global_logits.float()).abs().max().cpu()),
        float((control_output.local_logits.float() - candidate_output.local_logits.float()).abs().max().cpu()),
    )
    control.model.train(was_training[0])
    candidate.model.train(was_training[1])
    del volumes, present, metadata, position, control_output, candidate_output
    if max_abs > 1e-5:
        raise RuntimeError(f"B48 arms differ before a zero-start gate can open: max_abs={max_abs:.3e}")
    return max_abs


def run_b48_preflight(experiment: B48SubsetExperiment) -> dict:
    """Test B48's gradient boundary without a single optimizer step."""
    print(f"[{experiment.arm}] preflight: forward/backward only; no optimizer step")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = experiment.model
    model.train()
    batch = next(iter(experiment.train_loader))
    volumes, present, metadata, position, target = move_batch(batch)
    del batch
    saved_context_gate = model.sparse_head.context_gate.detach().clone()

    def local_loss_only() -> tuple[B48SubsetOutput, torch.Tensor]:
        with autocast_context():
            output = model(volumes, present, metadata, position, audit_context=True)
            loss = masked_bce_with_logits(output.local_logits, target, experiment.positive_weight)
        return output, loss

    model.zero_grad(set_to_none=True)
    with autocast_context():
        output = model(volumes, present, metadata, position, audit_context=True)
        combined_loss = masked_bce_with_logits(output.logits, target, experiment.positive_weight)
        local_loss = masked_bce_with_logits(output.local_logits, target, experiment.positive_weight)
        total_loss = combined_loss + experiment.config.local_loss_weight * local_loss
    experiment.scaler.scale(total_loss).backward()
    encoder_gradient = _has_nonzero_gradient(model.encoder.parameters())
    sparse_gradient = _has_nonzero_gradient([model.sparse_head.evidence_weight])
    if not encoder_gradient or not sparse_gradient:
        raise RuntimeError("Preflight failed: encoder and sparse evidence need gradients")
    model.zero_grad(set_to_none=True)

    # Local-only supervision may update spatial features and B48 gates, but not the global query branch.
    detached_output, detached_loss = local_loss_only()
    if detached_output.context_query.requires_grad:
        raise RuntimeError("B48 context query was not detached before local conditioning")
    experiment.scaler.scale(detached_loss).backward()
    gate_gradient = _has_nonzero_gradient([model.sparse_head.context_gate])
    leaked_global_gradient = any(parameter.grad is not None for parameter in model.global_branch.parameters())
    projections_still_closed = all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad).item())
        for parameter in (model.sparse_head.context_query.weight, model.sparse_head.context_key.weight)
    )
    if not gate_gradient or leaked_global_gradient or not projections_still_closed:
        raise RuntimeError("B48 zero-start detached-query gradient contract failed")
    model.zero_grad(set_to_none=True)

    # Opening the gate synthetically verifies that both low-rank projections become trainable later.
    with torch.no_grad():
        model.sparse_head.context_gate.fill_(0.05)
    opened_output, opened_loss = local_loss_only()
    experiment.scaler.scale(opened_loss).backward()
    projections_open = _has_nonzero_gradient(
        [model.sparse_head.context_query.weight, model.sparse_head.context_key.weight]
    )
    leaked_after_open = any(parameter.grad is not None for parameter in model.global_branch.parameters())
    with torch.no_grad():
        model.sparse_head.context_gate.copy_(saved_context_gate)
    model.zero_grad(set_to_none=True)
    if not projections_open or leaked_after_open:
        raise RuntimeError("B48 opened-gate or detachment contract failed")

    result = {
        "arm": experiment.arm,
        "total_loss": float(total_loss.detach().cpu()),
        "combined_loss": float(combined_loss.detach().cpu()),
        "local_loss": float(local_loss.detach().cpu()),
        "encoder_gradient": bool(encoder_gradient),
        "sparse_evidence_gradient": bool(sparse_gradient),
        "context_gate_gradient_at_zero": bool(gate_gradient),
        "context_projections_zero_at_zero_gate": bool(projections_still_closed),
        "context_projections_active_after_opening": bool(projections_open),
        "global_branch_isolated_from_local_loss": not leaked_global_gradient and not leaked_after_open,
        "host_peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "input_batch_gib": round(volumes.numel() * volumes.element_size() / 1024**3, 2),
    }
    if DEVICE.type == "cuda":
        result["cuda_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    del volumes, present, metadata, position, target, output, combined_loss, local_loss, total_loss
    del detached_output, detached_loss, opened_output, opened_loss
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))
    print(f"[{experiment.arm}] preflight: PASS")
    return result


def run_b48_pair_preflight(pair: B48MatchedPair) -> dict:
    """Run the no-update B48 check separately for both matched arms."""
    control, candidate = pair.arms()
    if control.train_uid_sha256 != candidate.train_uid_sha256:
        raise RuntimeError("Matched arms have different training split membership")
    if control.validation_uid_sha256 != candidate.validation_uid_sha256:
        raise RuntimeError("Matched arms have different validation split membership")
    if _model_fingerprint(control.model) != pair.initialization_fingerprint:
        raise RuntimeError("Control initialization changed before preflight")
    if _model_fingerprint(candidate.model) != pair.initialization_fingerprint:
        raise RuntimeError("Candidate initialization changed before preflight")
    zero_start_max_abs = check_zero_start_pair_equivalence(pair)
    return {
        "matched_initialization_fingerprint": pair.initialization_fingerprint,
        "zero_start_pair_max_abs_difference": zero_start_max_abs,
        "static_prior_control": run_b48_preflight(control),
        "post_cross_attention_candidate": run_b48_preflight(candidate),
    }


def run_b48_epoch(experiment: B48SubsetExperiment, loader: DataLoader, training: bool) -> dict:
    """Run one train/validation pass and preserve B48 context/top-k audit arrays."""
    experiment.model.train(training)
    losses, targets, probabilities, context_abs, overlaps = [], [], [], [], []
    for batch in loader:
        volumes, present, metadata, position, target = move_batch(batch)
        del batch
        if training:
            experiment.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context():
            output = experiment.model(
                volumes, present, metadata, position, audit_context=not training
            )
            combined_loss = masked_bce_with_logits(output.logits, target, experiment.positive_weight)
            local_loss = masked_bce_with_logits(output.local_logits, target, experiment.positive_weight)
            loss = combined_loss + experiment.config.local_loss_weight * local_loss
        if training:
            experiment.scaler.scale(loss).backward()
            experiment.scaler.unscale_(experiment.optimizer)
            torch.nn.utils.clip_grad_norm_(experiment.model.parameters(), experiment.config.grad_clip_norm)
            experiment.scaler.step(experiment.optimizer)
            experiment.scaler.update()
        losses.append(float(loss.detach().cpu()))
        targets.append(target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(output.logits).detach().cpu().numpy())
        context_abs.append(output.context_abs_mean.detach().cpu().numpy())
        if output.topk_overlap_with_static is not None:
            overlaps.append(output.topk_overlap_with_static.detach().cpu().numpy())
        del volumes, present, metadata, position, target, output, loss, combined_loss, local_loss
    return {
        "loss": float(np.mean(losses)),
        "target": np.concatenate(targets, axis=0),
        "probability": np.concatenate(probabilities, axis=0),
        "context_abs_mean": np.concatenate(context_abs, axis=0),
        "topk_overlap_with_static": None if not overlaps else np.concatenate(overlaps, axis=0),
    }


def train_b48_arm(experiment: B48SubsetExperiment) -> list[dict]:
    """Train one arm for the fixed subset duration using the common RNG seed."""
    # Reset stochastic layers for each arm; each private loader already has the same order seed.
    set_seed(experiment.config.seed + 97531)
    for epoch in range(1, experiment.config.epochs + 1):
        started = time.time()
        train_result = run_b48_epoch(experiment, experiment.train_loader, training=True)
        row = {"arm": experiment.arm, "epoch": epoch, "train_loss": train_result["loss"]}
        if experiment.validation_loader is not None:
            validation_result = run_b48_epoch(experiment, experiment.validation_loader, training=False)
            metrics = evaluate_predictions(validation_result["target"], validation_result["probability"])
            row.update(
                {
                    "validation_loss": validation_result["loss"],
                    "validation_mean_auc": metrics["mean_auc"],
                    "validation_known_cells": metrics["known_cells"],
                    "validation_context_abs_mean": float(validation_result["context_abs_mean"].mean()),
                    "validation_topk_change_fraction": (
                        None if validation_result["topk_overlap_with_static"] is None
                        else float(1.0 - validation_result["topk_overlap_with_static"].mean())
                    ),
                }
            )
        row["context_gate_abs_mean"] = float(
            experiment.model.sparse_head.effective_context_gate().detach().abs().mean().cpu()
        )
        row["fusion_gate_abs_mean"] = float(torch.tanh(experiment.model.fusion_gate).detach().abs().mean().cpu())
        row["elapsed_seconds"] = round(time.time() - started, 1)
        experiment.history.append(row)
        print(json.dumps(row, indent=2))
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return experiment.history


def train_b48_matched_pair(pair: B48MatchedPair) -> dict:
    """Train control first and candidate second; neither arm selects the other."""
    control, candidate = pair.arms()
    return {
        "static_prior_control": train_b48_arm(control),
        "post_cross_attention_candidate": train_b48_arm(candidate),
    }


def _evaluation_payload(experiment: B48SubsetExperiment) -> dict:
    """Run one fresh validation evaluation and serialize the arm's audit surface."""
    if experiment.validation_loader is None:
        raise ValueError("A validation split is needed for paired subset comparison")
    result = run_b48_epoch(experiment, experiment.validation_loader, training=False)
    metrics = evaluate_predictions(result["target"], result["probability"])
    per_target_context = dict(zip(TARGETS, result["context_abs_mean"].mean(axis=0).astype(float).tolist()))
    per_target_change = None
    if result["topk_overlap_with_static"] is not None:
        changes = 1.0 - result["topk_overlap_with_static"].mean(axis=0)
        per_target_change = dict(zip(TARGETS, changes.astype(float).tolist()))
    return {
        "validation_weighted_bce": result["loss"],
        "validation_metrics": metrics,
        "context_abs_mean_by_target": per_target_context,
        "topk_change_fraction_by_target": per_target_change,
        "effective_context_gate": dict(
            zip(
                TARGETS,
                experiment.model.sparse_head.effective_context_gate().detach().cpu().float().tolist(),
            )
        ),
        "effective_fusion_gate": dict(
            zip(TARGETS, torch.tanh(experiment.model.fusion_gate).detach().cpu().float().tolist())
        ),
    }


def evaluate_b48_matched_pair(pair: B48MatchedPair) -> dict:
    """Create the non-selective paired subset comparison and top-k change audit."""
    control, candidate = pair.arms()
    control_payload = _evaluation_payload(control)
    candidate_payload = _evaluation_payload(candidate)
    control_auc = control_payload["validation_metrics"]["per_target_auc"]
    candidate_auc = candidate_payload["validation_metrics"]["per_target_auc"]
    per_target_delta = {
        target: (
            None if control_auc[target] is None or candidate_auc[target] is None
            else float(candidate_auc[target] - control_auc[target])
        )
        for target in TARGETS
    }
    return {
        "scope": asdict(B48_SUBSET_REFERENCE),
        "matched_initialization_fingerprint": pair.initialization_fingerprint,
        "common_train_uid_sha256": control.train_uid_sha256,
        "common_validation_uid_sha256": control.validation_uid_sha256,
        "fixed_epochs": int(control.config.epochs),
        "static_prior_control": control_payload,
        "post_cross_attention_candidate": candidate_payload,
        "candidate_minus_control": {
            "validation_mean_auc": (
                None
                if control_payload["validation_metrics"]["mean_auc"] is None
                or candidate_payload["validation_metrics"]["mean_auc"] is None
                else float(
                    candidate_payload["validation_metrics"]["mean_auc"]
                    - control_payload["validation_metrics"]["mean_auc"]
                )
            ),
            "validation_weighted_bce": float(
                candidate_payload["validation_weighted_bce"]
                - control_payload["validation_weighted_bce"]
            ),
            "per_target_auc": per_target_delta,
        },
        "interpretation": (
            "Subset-only paired diagnostic; do not treat this as the official B48 scanner-domain result "
            "or use it for checkpoint/architecture selection."
        ),
    }


def plot_b48_pair_history(pair: B48MatchedPair) -> None:
    """Plot the matched arms' training and validation losses without selecting one."""
    plt.figure(figsize=(9, 4))
    for experiment in pair.arms():
        history = pd.DataFrame(experiment.history)
        if history.empty:
            raise ValueError("No completed epochs yet; train both B48 arms first")
        plt.plot(history["epoch"], history["train_loss"], marker="o", label=f"{experiment.arm} train")
        if "validation_loss" in history:
            plt.plot(
                history["epoch"],
                history["validation_loss"],
                marker="x",
                linestyle="--",
                label=f"{experiment.arm} validation",
            )
    plt.xlabel("epoch")
    plt.ylabel("masked weighted BCE loss")
    plt.title("B48-shaped matched subset arms (descriptive only)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_b48_pair_results(pair: B48MatchedPair, comparison: dict) -> None:
    """Display both histories and the compact paired comparison summary."""
    history = pd.concat([pd.DataFrame(experiment.history) for experiment in pair.arms()], ignore_index=True)
    display(history)
    summary = pd.DataFrame(
        [
            {
                "arm": arm,
                "validation_mean_auc": comparison[arm]["validation_metrics"]["mean_auc"],
                "validation_weighted_bce": comparison[arm]["validation_weighted_bce"],
                "topk_change_fraction_mean": None
                if comparison[arm]["topk_change_fraction_by_target"] is None
                else float(np.mean(list(comparison[arm]["topk_change_fraction_by_target"].values()))),
            }
            for arm in ("static_prior_control", "post_cross_attention_candidate")
        ]
    )
    display(summary)


def build_test_loader(paths: TestPaths, config: RunConfig = CONFIG) -> DataLoader:
    """Build a no-label test loader from the same extracted Drive subset."""
    validate_test_dataset(paths)
    test_table = pd.read_csv(paths.test_csv)
    series_table = pd.read_csv(paths.series_csv)
    test_table["StudyInstanceUID"] = test_table["StudyInstanceUID"].astype(str)
    records = build_series_records(series_table, config)
    test_table = test_table.loc[test_table["StudyInstanceUID"].isin(records)].reset_index(drop=True)
    if test_table.empty:
        raise ValueError("No test studies remain after matching metadata")
    dataset = KneeMRIDataset(test_table, records, paths, config, split="test", include_targets=False)
    print(f"test studies={len(dataset)}")
    return _make_loader(dataset, config, shuffle=False, seed=config.seed + 93)


def predict_test_set(experiment: B48SubsetExperiment, test_loader: DataLoader) -> pd.DataFrame:
    """Generate test probabilities for one arm without updating its model."""
    experiment.model.eval()
    fragments: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in test_loader:
            study_uids = list(batch["study_uid"])
            volumes, present, metadata, position = move_model_inputs(batch)
            with autocast_context():
                output = experiment.model(volumes, present, metadata, position)
            probability = torch.sigmoid(output.logits).float().cpu().numpy()
            fragment = pd.DataFrame(probability, columns=TARGETS)
            fragment.insert(0, "StudyInstanceUID", study_uids)
            fragment["predicted_positive"] = [format_positive_predictions(row) for row in probability]
            fragments.append(fragment)
            del batch, volumes, present, metadata, position, output
    predictions = pd.concat(fragments, ignore_index=True)
    print(f"[{experiment.arm}] generated predictions for {len(predictions)} test studies")
    return predictions


def save_b48_pair_results(
    pair: B48MatchedPair,
    comparison: dict,
    test_predictions: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Save both models, histories, comparison/audit, and optional test predictions to Drive."""
    run_name = time.strftime("b48_subset_matched_pair_%Y%m%d_%H%M%S")
    run_root = pair.static_prior_control.paths.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    for experiment in pair.arms():
        arm = experiment.arm
        torch.save(
            {
                "model_state": experiment.model.state_dict(),
                "config": asdict(experiment.config),
                "targets": TARGETS,
                "arm": arm,
                "context_source": experiment.model.context_source,
                "matched_initialization_fingerprint": pair.initialization_fingerprint,
                "b48_subset_scope": asdict(B48_SUBSET_REFERENCE),
            },
            run_root / f"{arm}_model.pt",
        )
        (run_root / f"{arm}_history.json").write_text(
            json.dumps(experiment.history, indent=2), encoding="utf-8"
        )
    (run_root / "config.json").write_text(
        json.dumps(asdict(pair.static_prior_control.config), indent=2), encoding="utf-8"
    )
    (run_root / "b48_subset_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (run_root / "b48_subset_scope.json").write_text(
        json.dumps(asdict(B48_SUBSET_REFERENCE), indent=2), encoding="utf-8"
    )
    if test_predictions is not None:
        for arm, prediction in test_predictions.items():
            prediction.to_csv(run_root / f"{arm}_test_predictions.csv", index=False)
    print("Saved matched B48 subset run to:", run_root)
    return run_root


# Retain the generic name used by the inherited case-review helper below.
Experiment = B48SubsetExperiment
''',
)

replace_cell(
    23,
    "markdown",
    "## 11. Build the matched B48 subset pair from the extracted Drive data",
)
replace_cell(
    24,
    "code",
    r'''
# Build two fresh arms from the same CSV split and byte-identical initial parameter tensors.
B48_PAIR = build_b48_matched_pair(PATHS, CONFIG)
''',
)
replace_cell(
    25,
    "markdown",
    "### 11a. Mandatory no-update B48 detachment, zero-gate, and memory check",
)
replace_cell(
    26,
    "code",
    r'''
# Both arms must pass before either optimizer is allowed to take a step.
B48_PREFLIGHT = run_b48_pair_preflight(B48_PAIR)
''',
)
replace_cell(
    27,
    "markdown",
    "### 11b. Train both fixed-duration arms, compare descriptively, predict, and save",
)
replace_cell(
    28,
    "code",
    r'''
# Keep both optimizers off until the preflight cell reports PASS for both arms.
RUN_B48_TRAINING = False

# Enable this one switch only when the matched preflight has passed.
if RUN_B48_TRAINING:
    # Train both arms for the same fixed number of epochs; no arm is selected here.
    B48_HISTORY = train_b48_matched_pair(B48_PAIR)
    # Plot both loss trajectories together for descriptive inspection.
    plot_b48_pair_history(B48_PAIR)
    # Build the explicit paired subset comparison plus per-target top-k-change audit.
    B48_COMPARISON = evaluate_b48_matched_pair(B48_PAIR)
    # Display histories and the compact arm-level subset summary.
    show_b48_pair_results(B48_PAIR, B48_COMPARISON)
    # Build the same local-DICOM test loader used by the earlier notebook.
    TEST_LOADER = build_test_loader(TEST_PATHS, CONFIG)
    # Keep one test prediction table per matched arm rather than blending them.
    B48_TEST_PREDICTIONS = {
        experiment.arm: predict_test_set(experiment, TEST_LOADER)
        for experiment in B48_PAIR.arms()
    }
    # Review twelve unlabelled test cases from the candidate arm only for visualization.
    TEST_CASE_TABLE = show_case_examples(
        B48_PAIR.post_cross_attention_candidate,
        loader=TEST_LOADER,
        max_cases=12,
        title_prefix="Candidate test",
    )
    # Save both new models, histories, comparison JSON, scope JSON, and arm-specific predictions.
    RUN_DIRECTORY = save_b48_pair_results(
        B48_PAIR,
        B48_COMPARISON,
        test_predictions=B48_TEST_PREDICTIONS,
    )
''',
)
replace_cell(
    29,
    "markdown",
    r"""
## Memory controls and paired-run discipline

The defaults stream DICOM data, retain only four series per study, encode one
triplet at a time, and checkpoint the image encoder. The compact global
series-memory branch adds little memory compared with the 448×448 image path.
If either arm fails preflight, change only one shared `CONFIG` setting, rebuild
`B48_PAIR`, and rerun both preflights:

1. Reduce `max_series_per_study` from `4` to `3`.
2. Keep `encoder_chunk_size=1` and `gradient_checkpointing=True`.
3. Reduce `slices_per_series` from `32` to `24` only as a last resort; this
   changes the representation for both arms.
4. Keep `image_size=448` and `resize_policy="aspect_preserving_pad"` unless you
   intentionally want a different common representation.

Do not run just one arm, alter a setting between arms, inspect the first result
and then tune the second, or treat the subset comparison as the official B48
decision. The notebook records split hashes, the common initialization
fingerprint, context-gate values, and per-target top-k changes precisely to make
this sandbox comparison easy to audit without overstating it.
""",
)


if __name__ == "__main__":
    build(Path(__file__).with_name("b48_global_conditioned_sparse_mil_colab.ipynb"))
