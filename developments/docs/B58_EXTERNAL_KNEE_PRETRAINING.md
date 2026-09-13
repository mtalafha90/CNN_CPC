# B58: adapt DINOv2 to additional knee MRI images

**Status (2026-09-13): implemented; no real-data B58 training or AUC yet.**
The user reports the B57 DINOv2 arm is running on the 5090. B58 is a separate
A4500 experiment. Keep the running 5090 checkout and environment unchanged.
All A4500 commands run from `/media/talafha/Disk_1/CNN_CPC`; outputs stay in
`runs/094_Experiment_B58_external_knee_pretraining/`.

The question is whether a short adaptation stage using **RSNA training images
plus MRNet, fastMRI and OAI** improves the B57 DINOv2 diagnosis model. External
images teach the encoder MRI structure without diagnosis labels. Then the
original RSNA labels train the same twelve outputs using B57's recipe.
Adding images does not guarantee a higher AUC.

This compares complete recipes: B58 adds both self-supervised optimization and
external images. A gain cannot be attributed to external data alone. A separate
RSNA-only adaptation control would be needed for that causal claim. The direct
reference is the completed **B57 DINOv2 candidate**, not the historical B52
checkpoint and not B57's ConvNeXt arm. B58 does not need that ConvNeXt run to
finish before this comparison.

## Data to obtain

Obtain and extract the provider releases using their access procedures. The
launcher does not download MRI datasets or accept provider agreements. It
requires all three external sources and stops if any is absent.

| Source | Required files under `CNN_CPC` | B58 use |
|---|---|---|
| Original RSNA | Existing `rsna-knee-abnormality-detection`, original teacher export, series policy and B50 gate | Allowed training images in adaptation; original labels in diagnosis training |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | `data/external/mrnet/train/{axial,coronal,sagittal}/*.npy`; an intervening `MRNet-v1.0` folder also works | Official training partition only; no MRNet labels, validation or test images |
| [fastMRI knee](https://fastmri.med.nyu.edu/) | `data/external/fastmri/knee_multicoil_train/*.h5` | The real-valued `reconstruction_rss` image volumes from the knee multicoil training release |
| [OAI](https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative) | Extracted knee MRI DICOM series below `data/external/oai/` | Unlabelled knee MRI; visits and knees grouped by the DICOM participant ID |

fastMRI's [official format documentation](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md)
distinguishes complex k-space from reconstructed image targets. B58 consumes
`reconstruction_rss`, not complex k-space or challenge test files. Download the
**knee** training release; a brain release is not an alternative input.
The generic official folder name `multicoil_train` is also accepted.

For OAI, use extracted MRI DICOM files, including extensionless DICOM files.
X-rays, clinical tables and unopened ZIP archives are not training images.
Scouts, localizers, thigh images and series with fewer than three slices are
excluded. Eligible MR series require knee/DESS/IW/T1/T2 identifiers in their
headers and stable participant, study and series IDs. Enhanced multiframe
files, repeated slice positions, mixed matrices and decoding errors fail
explicitly. Physical slice positions are used when present, with
`InstanceNumber` as the fallback. Do not repair images by renaming files or
removing failed slices without first investigating the source.

External labels are never mapped to missing RSNA diagnoses. The sources do
not all supply the same twelve labels. Before a later competition submission,
check the competition's current external-data rules and the relevant release
terms; this implementation is a research training path, not an eligibility
determination.

## Frozen design and A4500 budget

The machine-readable recipe is
[`config/b58_external_knee_pretraining.json`](../../config/b58_external_knee_pretraining.json).
The implementation lives in the new `rsna_knee.b58_external_knee` subpackage;
the B57 source modules and configs are unchanged.

| Item | B58 v1 |
|---|---|
| Initial encoder | Exact B57 public DINOv2-S/14 weights, checked with the existing tensor fingerprint |
| External selection | Seeded hash selection, at most 1,024 groups per source and two series per group |
| Group identity | RSNA study; MRNet exam; fastMRI acquisition; OAI participant |
| Adaptation sampling | Repeating source cycle: RSNA, MRNet, RSNA, fastMRI, RSNA, OAI; uniform group, then series, then cached centre |
| Pixel cache | Eight centres per selected series, three adjacent slices per centre, float16, 224-square views |
| Adaptation | 2,000 optimizer steps; batch eight; seed 2026 |
| Learning | Last four encoder blocks plus final norm and a fresh projection head; earlier blocks frozen |
| Objective | Two-view DINO-style self-distillation, centered teacher targets and an EMA teacher |
| Optimizer | AdamW; encoder LR `1e-5`, projector LR `1e-4`, decay `0.04`; 100-step warmup and cosine decay |
| Teacher | Momentum `0.996` increasing to one; teacher/student temperatures `0.04` / `0.1`; 2,048 prototypes |
| Diagnosis training | Fresh B57 heads, adapted encoder, all encoder blocks trainable, unchanged B57 twelve-epoch schedule |
| Endpoint selection | Fixed final EMA encoder, then fixed final diagnosis epoch; no best-epoch selection |
| Recovery | Adaptation every 100 steps; diagnosis training after each complete epoch |

This is a small adaptation of the
[DINO self-distillation method](https://github.com/facebookresearch/dino),
starting from [DINOv2](https://github.com/facebookresearch/dinov2). It is not a
reproduction of full DINOv2 pretraining: it has no local views, masked-patch
objective or large distributed batch. Teacher entropy and probability
variation are logged so representation collapse can be investigated.

The adaptation cache uses volume-level intensity normalization, a 90% central
crop, isotropic resize and edge padding. Each pair of views has independent
80–100% area crops, rotation within seven degrees, mild intensity changes and
noise; each view applies the same transform across its three slice channels.
The maximum cache estimate is about **18.4 GiB**, plus checkpoint space. Raw
downloads can be much larger. The exact selected subset and cache estimate
print before caching. The experiment does not claim to consume every image in
these datasets.

Diagnosis training keeps B57's original 32-centre, 448-area rectangular B42
preprocessing, series aggregation, target masks, original teacher, optimizer,
seed and twelve-epoch budget. It uses the full allowed RSNA training set;
the 1,024-group cap applies only to self-supervised adaptation. Mixed precision,
encoder chunking and gradient checkpointing follow the existing runtime.
Actual A4500 memory use must pass both real-data backward preflights.

## Data boundary and reproducibility

B58 builds its own RSNA protocol with the existing B57 preparation code. It
reads the existing B50 gate and original teacher; it does not recreate the gate
or modify a B57 run directory. With identical input artifacts, the previously
reported B57 population is 3,603 train / 548 validation / 198 excluded-profile
studies; preparation records the actual counts on this machine.

RSNA validation, Expert-58 and excluded-profile rows never enter adaptation
or diagnosis gradients. MRNet and fastMRI official validation/test partitions
are not indexed. OAI is checked against RSNA study and series identifiers;
identical cached content anywhere in the selected pool is refused. These
checks **do not certify cross-dataset patient independence**: MRNet/fastMRI
identifiers cannot establish identity across unrelated releases, and resized
or re-exported copies can evade exact-content checks. The protocol records
that limitation. No independent external validation cohort is claimed.

Preparation freezes source paths, selected groups/series, raw file hashes,
cache hashes, config, implementation digest and RSNA artifacts. Training
verifies the cache, and each sampled cache file is checked before use.
Changing source, data, config or runtime during recovery is refused. Keep the
A4500 checkout and environment fixed once B58 is prepared. Completed steps
verify and return without replacing their final model.

## Commands on the A4500

First download/extract the three external sources into the folders above and
make the existing RSNA inputs available on this machine. Then, in the A4500
terminal only:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
(
  set -euo pipefail
  git pull --ff-only origin main
  python -m pip install -e '.[test,b58]'
  python -m pytest -q developments/tests/test_b58_external_knee.py
  bash developments/scripts/run_b58_external_knee.sh prepare
  bash developments/scripts/run_b58_external_knee.sh run \
    2>&1 | tee -a runs/094_Experiment_B58_external_knee_pretraining/train.log
)
```

The subshell stops on a failed command without closing the interactive shell.
`prepare` indexes, hashes and caches real MRI images; it may take substantial
CPU and disk time. `run` executes both zero-step preflights, the 2,000-step
adaptation, then an adapted-encoder preflight and the twelve diagnosis epochs.
The stages use separate processes so one stage's models do not occupy memory
during the next stage. There is no automatic comparison or submission.

Defaults use the existing original-teacher and series-policy paths:

```text
runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all
runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json
```

If your folders differ, set the matching `B58_DATA_ROOT`, `B58_LABELS_ROOT`,
`B58_SERIES_POLICY`, `B58_SELECTION_ROOT`, `B58_MRNET_ROOT`, `B58_FASTMRI_ROOT`
or `B58_OAI_ROOT` before preparation. The gate is automatically discovered
under the primary `runs/` only when exactly one existing B50 gate is found.
All these paths also have explicit CLI options; use
`bash developments/scripts/run_b58_external_knee.sh --help`.

After an interruption, use the same checkout and environment and run:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
bash developments/scripts/run_b58_external_knee.sh run
```

This resumes the most recent adaptation checkpoint or completed diagnosis
epoch. It does not require another download or a regenerated protocol.

## Compare after both machines finish

Once the 5090 prints completion for `dinov2_slice_candidate`, copy these
finished B57 artifacts, preserving their relative structure, into
`CNN_CPC/runs/imported_b57_5090/` on the A4500:

```text
protocol/                         entire directory
dinov2_slice_candidate/final.pt
dinov2_slice_candidate/complete.json
dinov2_slice_candidate/validation.npz
dinov2_slice_candidate/validation.json
dinov2_slice_candidate/expert.npz
dinov2_slice_candidate/expert.json
```

Do not copy an active or incomplete checkpoint as the reference. B58 itself
does not depend on these files until comparison:

```bash
cd /media/talafha/Disk_1/CNN_CPC
bash developments/scripts/run_b58_external_knee.sh compare \
  --reference-root "$PWD/runs/imported_b57_5090"
```

The reader verifies the exported B57 protocol and prediction hashes, while
allowing that machine's old mount paths and source digest. The semantic
comparison still requires identical RSNA input bytes, labels, confidence
masks, study populations, scanner profiles and supervised recipes. Prediction
rows are aligned by study ID and checked against the frozen labels. Different
Torch/precision/library versions are reported as runtime differences; the
comparison is not claimed to be bitwise identical across GPU architectures.

`comparison_with_b57.json` contains both macro and per-target AUCs and paired
scanner-cluster bootstrap intervals. Development support requires validation
delta at least **+0.010** and a positive 95% lower bound. Expert-58 is diagnostic
only. This reused development result needs independent confirmation before
promotion. B58 does not select a blend or automatically submit to Kaggle.

## Local verification

Focused tests cover official-training partition selection, actual NPY/HDF5/
DICOM pixels, physical ordering, participant grouping, missing-source and
overlap failures, float16 cache integrity, sampler recovery, real DINOv2
gradients and strict encoder transfer. Both full training loops are exercised
on synthetic data with interrupted/resumed checkpoints and exact final-weight
comparison; prediction comparison checks frozen labels. The existing B57
suite remains a regression gate. These CPU checks do not replace the A4500
real-data preflights or establish an AUC result.
