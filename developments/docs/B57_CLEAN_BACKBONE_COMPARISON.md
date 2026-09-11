# B57: a clean reference and a different MRI representation

**Status: implemented; no competition-data training or AUC result yet.**
The protocol is `config/b57_clean_backbone_comparison.json`. Run everything
from the primary `CNN_CPC` checkout. Outputs go to
`runs/093_Experiment_B57_clean_backbone_comparison/`. No sibling worktree is
created. The 093 directory avoids the existing 091/092 teacher artifacts;
the historical archive registry ending at B50 is not a current-run allocator.

## What B57 combines

| Stage | Purpose |
|---|---|
| Data and ancestry audit | Exclude validation studies from every competition training stage, including initialization; hash inputs and scanner assignments |
| Clean B52-style reference | Train the existing B42/B50 architecture from public ConvNeXt-Tiny weights with fresh hierarchy and sparse heads |
| DINOv2 slice candidate | Fully fine-tune DINOv2-S/14, contextualize slices within each series, and aggregate all eligible series into 12 predictions |
| Shared AUC evaluation | Compare identical subjects, labels and masks; report per-target coverage and paired uncertainty |
| Fixed ensemble diagnostic | Evaluate one 50:50 probability average; retain only if it passes a declared development gate |

This is a comparison of two complete model recipes, not a capacity-matched
ablation of one layer. The candidate changes the public pretraining source,
encoder, aggregation and auxiliary-loss design. The comparison can justify a
better recipe; it cannot attribute a gain to one component.

B55's regraded teacher and physical crop are not included in v1: B55 lost to
B52 on both recorded teacher surfaces. B56 remains the separate geometry
experiment documented in [its runbook](B55_RUNBOOK.md). A larger grid, tiles,
extra TTA, calibration, per-target weights and a teacher sweep are also absent.
Existing B52/B53/B55 code and checkpoints are preserved.

## Why initialization changes

B52's local 548-study validation set was held out from its fine-tuning loop,
but those subjects were already exposed to supervised ancestor training:

1. `b52_competition_training.train_b52` loads a Phase-9 `llm_fill` checkpoint
   before choosing its training/validation indices.
2. `b35_training._require_base_checkpoint` requires a completed two-epoch
   ancestor exposing all 4,349 report-only studies, 34,010 supervised cells,
   and a trainable encoder tail.
3. `phase9_matched_supervision_training.train_phase9_arm` builds its loader
   over all those report-only UIDs and checks every epoch's exposure counts.

Consequently, `0.834998` is not an independent held-out estimate for the B52
lineage. This does **not** invalidate B52's recorded Kaggle `0.716`, establish
how large the validation bias is, or prove that leakage caused the plateau.
It means B57 needs its own clean reference; comparing its AUC directly to
`0.834998` would mix protocols.

The B57 training CLI accepts no ancestor checkpoint. Only the exact public
encoder tensor fingerprints in `b57_models.py` are accepted; all competition
heads start fresh. Public source URLs, downloaded artifact hashes and package
versions are recorded. No Phase-9, B16 report-aligned, B52 or other competition
weights are imported. The provenance claim covers this code path; it is not an
independent audit of the public pretraining datasets.

## Data contract

Use the original 34,010-cell report teacher and the existing B50 gate. The
gate JSON hash, original `train.csv` hash and original teacher hash must match.
The gate is read, never regenerated. Freeze a separate B57 protocol before
either arm starts. Every later invocation verifies its inputs and source code.

The validation UIDs remain the gate's `validation_unseen_scanners` subjects
(548 in the recorded full-data runs). Training uses other report-only studies
**only when their scanner profile is absent from validation**. This extra
check covers the old `excluded_prior_surface` rows as well. The actual count
is printed at preparation; it may be less than B52's 3,801. Excluded profile
overlaps are neither trained on nor added to validation. Both training and
validation must contain a positive and a negative label for each target.

These validation studies have been examined in past experiments. They are a
**reused development set**, not a fresh independent test. Scanner profiles
describe manufacturer/model/field-strength groups; they are not patient or
hospital identities. Patient identity separation beyond StudyInstanceUID is
not certified by the available contract. An additional group-disjoint fold
with both arms restarted from public weights is required before promotion.

Target values, confidence weights, train-only target balancing and all-series
eligibility are identical between arms. The 58 expert studies receive no
gradients and are evaluated only after each fixed final epoch. Their small,
reused diagnostic cannot select a checkpoint, mixture or teacher. Coverage
counts expose missing report supervision; B57 does not manufacture new
image-reviewed labels or discard difficult positives to improve a score.

## Model and training recipe

Both arms use the established B42 full-volume normalization, native 90% crop,
aspect-preserving area equivalent to 448², 32 deterministic 2.5D slice centres,
and all eligible sagittal/coronal/axial series. Strict DICOM decoding prevents
silent missing-image predictions. Augmentation is explicitly disabled in both
arms; this is the actual pixel behaviour of the successful B52 run.

The reference reconstructs the B34/B50 hierarchy with fresh parameters, then
adds the B42 6×6/top-8 sparse head. All encoder stages and inference hierarchy
parameters train. The historical training-only local-context scaffold stays
bypassed and frozen. Fresh hierarchy parameters use the head learning rate,
rather than the small rate intended for an already trained hierarchy.

The candidate uses the authors' DINOv2-S/14 weights through **timm 1.0.20**.
Each triplet produces a 384-dimensional CLS feature. A single 6-head
Transformer with a series CLS token summarizes the slices. Plane/fluid/fat
metadata identify series, and 12 learned pathology queries attend across
series. Acquisition list order is arbitrary. Following the MST positional
ablation, there is no extra slice-position embedding: this tests slice-context
aggregation, **not a claim of physical-spacing or order-sensitive reasoning**.
The ViT adds reflection padding of at most 13 pixels per dimension for patch-14
compatibility; it does not resize every rectangle to a square.

The public encoders contain 27,820,128 reference and 22,056,192 candidate
parameters. Fresh trainable heads contain 19,003,428 and 1,790,220 respectively.
Both train all encoder weights. Chunk size 2 and encoder gradient checkpointing
limit activation memory; real GPU memory and runtime must be measured locally.

The fixed recipe is seed 2026, **12 complete epochs per arm**, effective batch
2, AdamW, encoder LR `1e-5`, fresh-head LR `1e-4`, weight decay `1e-4`, gradient
clip 1, and cosine decay to 1% of each initial LR. Each allowed training study
appears once per epoch. Reference loss is combined BCE plus auxiliary sparse
BCE; the candidate uses study BCE. Both use the same train-only target balance.

The primary endpoint is the **final epoch**, chosen before any B57 result.
Per-epoch AUCs are logged but no best epoch is substituted. Recovery saves the
optimizer, scheduler, AMP scaler, RNG state and complete data/model contract
atomically. Epoch-indexed shuffling reproduces the next epoch after restart.
Changed labels, code, split, model arm, initialization, schedule, precision or
recorded package versions are refused on resume. Keep this checkout and its
inputs fixed until both arms finish.

## Evaluation and gates

`comparison.json` contains both model AUCs, the one equal-probability ensemble,
per-target deltas, positive/negative/unlabelled counts, and paired bootstrap
intervals. Validation bootstraps resample scanner profiles together for both
models. Expert diagnostics resample studies. Replicates losing a measurable
target are counted and skipped; insufficient valid replicates yield no interval.
These intervals describe this reused surface, not Kaggle generalization.

- Candidate development support: macro delta at least `+0.010`, and paired
  95% interval lower bound above zero.
- Ensemble development support: candidate no more than `0.002` below the
  reference, ensemble delta at least `+0.005`, and interval lower bound above zero.
- Expert scores cannot trigger either gate. No automatic promotion, blending
  search or Kaggle submission occurs. A failed gate is recorded without changing
  the recipe to chase the result.

These thresholds are project decisions, not gains predicted by a paper.

## Run from the main project directory

After the current GPU run finishes:

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e '.[test,b57]'
python -m pytest -q developments/tests/test_b57_clean_comparison.py
bash developments/scripts/run_b57_clean_comparison.sh preflight
```

`preflight` first freezes the protocol and downloads the two public encoders
(roughly 200 MB total), then runs forward/backward checks on the two training
studies with the most eligible series for each arm. It makes **zero optimizer
steps**, verifies encoder and head gradients, and records peak GPU allocation.
This is a sample-based memory check, not a guarantee that every native image
shape fits. Training and evaluation subsequently use only the local weights.

The launcher discovers the existing B50 gate under `CNN_CPC/runs`. If there is
more than one distinct gate, set `B57_SELECTION_ROOT` to the intended existing
directory; do not regenerate or delete one. Other optional overrides are
`B57_DATA_ROOT`, `B57_LABELS_ROOT`, `B57_SERIES_POLICY`, `B57_RUN_ROOT` and
`B57_NUM_WORKERS` (default 0 for the A4500 laptop setup). Its default labels
path is the original 067 teacher, not a possibly stale shell `LABELS_ROOT`.

Once both preflights pass:

```bash
set -o pipefail
bash developments/scripts/run_b57_clean_comparison.sh train \
  2>&1 | tee -a runs/093_Experiment_B57_clean_backbone_comparison/train.log
bash developments/scripts/run_b57_clean_comparison.sh compare
```

The arms run sequentially in separate processes. Repeating `train` resumes an
interrupted arm at the last complete epoch and skips a verified completed arm.
Alternatively `run` executes prepare, preflights, training and comparison in
sequence. There is no claim that two 12-epoch runs fit a Kaggle time limit.

Outputs include `protocol/`, `public_init/`, both preflight JSONs, and per-arm
`final.pt`, `recovery_latest.pt`, `history.json`, validation/expert predictions
with provenance, and `complete.json`. B57 checkpoints are self-contained for
model reconstruction with `build_model(..., public_root=None)` plus strict
`model_state` loading. Existing B52 submission loaders cannot load them; a
Kaggle-specific runtime/package is a later step if the evidence supports it.

## Evidence behind the candidate

[Medical Slice Transformer](https://www.nature.com/articles/s41598-025-09041-8)
adapts 2D DINOv2 features to volume diagnosis using a slice Transformer. Its knee
experiment concerns meniscus classification on MRNet, not this competition's
12 targets; its reported AUC is not a B57 prediction. B57 is an adaptation to
this repository's multi-series inputs and label contract.

The [official DINOv2 repository](https://github.com/facebookresearch/dinov2)
provides the public backbone. The
[timm model card](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
documents the 384-dimensional slice features and LVD-142M pretraining. The
download/conversion path and full tensor fingerprints were checked locally.
