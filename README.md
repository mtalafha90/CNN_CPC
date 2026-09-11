# CNN_CPC — knee MRI abnormality detection

Twelve binary findings per knee MRI study, scored as macro ROC AUC.

## Latest competition work

The best recorded Kaggle endpoint is **B52 full-data, 0.716**. B42 remains the
preserved operational reference below. See the
[B52 submission record](developments/docs/B52_KAGGLE_SUBMISSION.md).

**B57 is implemented and unrun:** a clean B52-style reference, a fully
fine-tuned DINOv2 slice model, shared label/validation checks and a fixed
ensemble diagnostic. Start with the
[B57 runbook](developments/docs/B57_CLEAN_BACKBONE_COMPARISON.md).
It runs inside `CNN_CPC/runs/093_Experiment_B57_clean_backbone_comparison`.
Its reused development split is held out from the entire new training lineage;
old B52 local AUCs are not directly comparable to this clean protocol.

## Current operational endpoint

**B42 constant-area native-aspect sparse MIL** is the maintained operational
baseline. It preserves in-plane aspect ratio, uses ragged rectangular series
tensors, and has a successful hidden Kaggle result of **0.714**—tied with B37
and B41. This is an operational reference decision, not a claim that B42 is
scientifically superior to the tied endpoints.

Start with [`docs/ACTIVE_ENDPOINTS.md`](docs/ACTIVE_ENDPOINTS.md). It records
the exact B42 config, checkpoint fingerprint, inference path, required
external artefacts, and the status of closed B46--B49 experiments.

The generic top-level B34/224-pixel commands remain for backward compatibility
and historical reproduction. They are **not** the current Kaggle baseline and
must not be used accidentally for a new submission.

The B42 implementation lives in `developments/src/rsna_knee/` because it was
created after the earlier repository restructuring. It is now included in the
editable package, alongside the compatibility interface.

## Structure

```text
CNN_CPC/
├── config/                         frozen endpoint configurations
├── developments/src/rsna_knee/     active research and submission code
├── developments/docs/              governed experiment records
├── model/, data/, training/, ...   legacy B34 compatibility interface
├── docs/ACTIVE_ENDPOINTS.md        maintained endpoint registry
├── docs/ENVIRONMENT.md             local and offline-Kaggle runtime contract
├── tests/                          compatibility-interface tests
├── runs/                           external, immutable run artefacts
├── requirements.txt
└── pyproject.toml
```

`developments/` is a preserved lineage, but B42's maintained endpoint code is
currently located there. Treat its experiment documents as the scientific
source of truth; [`developments/docs/CURRENT_STATUS.md`](developments/docs/CURRENT_STATUS.md)
is the living status record.

## Installation

```bash
conda activate rsna-knee
pip install -e .
```

This installs both the compatibility packages and `rsna_knee`, so the B42
loader and frozen Kaggle inference modules are importable. The dataset,
checkpoints, label artefacts, and run outputs are deliberately not stored in
Git.

## B42 reference material

The B42 model, fixed-E2 training contract, and geometry are documented in
[`developments/docs/B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md`](developments/docs/B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md).
Its hidden-safe dual-T4 submission implementation is
`rsna_knee.b42_constant_area_aspect_sparse_submission_dualgpu_fast`.

Do not use a generic training or submission command as a substitute for that
frozen contract. The endpoint must verify its exact checkpoint and base-model
fingerprints before prediction.

## Two things to keep in mind

**The 58 expert-annotated studies are not a model-selection test set.** They
remain useful as a diagnostic only. B57 declares a reused development comparison;
an additional group-disjoint confirmation is required before promotion.

**B46, B48, and B49 are closed.** Their results do not authorize a new gold
weight, tile geometry, crop, query, calibration, blend, or seed search. See
[`developments/docs/CURRENT_STATUS.md`](developments/docs/CURRENT_STATUS.md).

## Tests

```bash
pip install -e ".[test]"
python -m pytest
```

Covers the compatibility interface and the selected active B42 regression
surface in CI.
