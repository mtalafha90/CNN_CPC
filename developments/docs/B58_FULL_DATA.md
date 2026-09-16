# B58 full-data and native-label run (v2)

Implemented 2026-09-16 at the user's request. No real-data A4500 training or
AUC result is recorded. This revision replaces the proposed small B58 run for
new work; the original v1 implementation and frozen artifacts remain usable.

## What runs

| Stage | Data | Supervision | Fixed schedule |
|---|---|---|---|
| Image adaptation | Every eligible training series from RSNA, MRNet, fastMRI knee and OAI | Two-view student/EMA-teacher learning, including unlabelled cases | 1 complete series pass; 8 sampled triplets per series at 224 pixels |
| Native diagnosis training | Every labelled training case, with all its eligible series | Shared DINOv2-S encoder and separate source-native heads | 6 complete case passes; 32 triplets per series at 224 pixels |
| RSNA fine-tuning | Every study in the frozen RSNA training partition | Original report-label targets and confidence masks, fresh B57 study heads | The unchanged 12-epoch B57 recipe, including its rectangular 448-area preprocessing |

The public DINOv2 initialization remains pinned. All twelve encoder blocks are
trainable during the new adaptation stages. The final EMA encoder from image
adaptation initializes native diagnosis training; its final encoder initializes
RSNA fine-tuning. The trained native heads are also saved in `native_model.pt`.
The final RSNA model produces the same twelve competition probabilities.

There is **no 1,024-group cap, two-series cap, or 2,000-step cutoff**. Step totals
come from the frozen inventory. An epoch visits every scheduled item exactly
once, with deterministic within-source shuffling and interleaving. Native
losses use inverse source frequency to equalize each source's aggregate loss
coefficient over a pass, before gradient clipping; this does not make optimization
independent of source order or dataset size. SSL uses the actual series frequencies.

“Full” means all eligible data in the supplied training inventories. It does
not mean every voxel on every update, unprovided releases, or held-out data.
MRNet specifically requires all 1,130 official training exams and three planes.
The prepare report gives exact installed fastMRI/OAI counts; it cannot certify
that the entire worldwide release has been downloaded. Scouts, thigh images,
series shorter than three slices, unsupported enhanced multiframe DICOM, and
invalid volumes are excluded or refused by the documented readers. Training
does not silently skip corrupt images.

RSNA validation/expert/excluded scanner profiles never enter gradients at any
stage. MRNet validation/test and fastMRI validation/test are not indexed. OAI
requires an explicit participant-disjoint partition map; all visits and knees
of a participant must stay in one partition. Labels may cover a larger OAI
release than the downloaded images; unmatched rows are reported, not claimed
as used. Cross-source patient identity cannot be certified from these releases;
RSNA/OAI UID overlap and exactly duplicated training volumes are refused.

## Which labels are learned

| Source | Native targets | Missing-label behavior |
|---|---|---|
| RSNA | Original 12 report findings, soft targets and confidence weights | Original zero-weight cells remain masked |
| MRNet | Abnormal, ACL tear, meniscus tear | All three binary labels required for every training exam |
| fastMRI+ | All finding names in the provided knee annotation release, including its artifact category | Annotated findings are positive; reviewed scans with no findings are negative for the release vocabulary; other absent findings and unreviewed scans are unknown |
| OAI | Every declared, joined native task: binary finding, categorical grade or bounded continuous score | Empty cells and explicitly declared missing codes receive zero weight |

MRNet's generic meniscus label is **not** relabelled as medial or lateral.
The fastMRI+ boxes are aggregated into acquisition-level finding presence for
this classification model; this run does not train a bounding-box detector.
The OAI heads retain their original task meaning. Categorical grades use
cross-entropy; bounded continuous scores are scaled by declared dictionary
bounds and use sigmoid/Smooth-L1; binary tasks use masked BCE. There is no
automatic conversion of OAI scores to RSNA diagnoses. Supervised cases pool
slice features within each series, then series features within the case.

The label decisions follow the primary dataset descriptions:

- [Stanford MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/): three exam labels and patient-disjoint official partitions.
- [fastMRI+ authors' repository](https://github.com/microsoft/fastmri-plus): `knee.csv` plus `knee_file_list.csv`; labels are non-exhaustive, and reviewed files without annotations have no recorded findings.
- [OAI data portal](https://nda.nih.gov/oai/): use the actual downloaded release's clinical/imaging assessments and its dictionary. No universal OAI diagnosis CSV schema is assumed here.

## Files you must have

Use the main checkout `/media/talafha/Disk_1/CNN_CPC`. No sibling worktree is
needed. Images remain where downloaded; only outputs go under:

```text
CNN_CPC/runs/094_Experiment_B58_external_knee_pretraining/full_data_v2/
```

Keep the existing RSNA dataset, original teacher export, series policy, and
frozen B50 gate used by B57. This run prepares a separate local RSNA protocol.
The configuration is `config/b58_full_data.json`; freeze it before starting.
Do not change it mid-run or repurpose old v1 recovery files.

External input layouts:

```text
MRNet/
  train/axial/0000.npy ... 1129.npy
  train/coronal/0000.npy ... 1129.npy
  train/sagittal/0000.npy ... 1129.npy
  train-abnormal.csv
  train-acl.csv
  train-meniscus.csv

fastmri/
  knee_multicoil_train/*.h5
fastmri-plus/Annotations/
  knee.csv
  knee_file_list.csv

oai/
  ... extracted original knee DICOM series ...
oai_labels/
  series.csv
  labels.csv
  tasks.json
```

The fastMRI HDF5 files must contain `reconstruction_rss`. Raw k-space alone
does not supply diagnoses; obtain the separate fastMRI+ annotations.

MRNet CSVs normally have headerless `exam_id,0_or_1` rows. Recognized named
headers are also accepted. If Redivis promoted `0000,1` to `_0000,_1` column
names, the importer restores that exact observation and logs it. A 1,129-row
export without that encoded observation fails instead of guessing the label.

### Join the OAI release explicitly

1. Export an identity inventory from the actual DICOMs:

   ```bash
   bash developments/scripts/run_b58_full_data.sh inventory-oai \
     --oai-root "$B58_OAI_ROOT" --oai-series "$B58_OAI_SERIES"
   ```

   This refuses to overwrite an existing map. It includes patient, study and
   series identifiers, acquisition-date/laterality hints, and an example path.
   `visit`, `side`, and `partition` are deliberately blank: join them from the
   official image-release metadata. A date must not be guessed to be a visit.

2. Complete the map. Required columns are
   `series_uid,patient_id,visit,side,partition`. Use side `L` or `R`, and partition
   `train`, `validation`, or `test`. Every eligible loaded series must be mapped;
   references to missing/ineligible series fail. Participant IDs must match
   DICOM PatientID; known DICOM laterality must agree. Preserve any existing
   evaluation partition rather than assigning those participants to training.

3. Produce `labels.csv` from the clinical/assessment release, one row per
   `patient_id,visit,side`, followed by its native target columns. Retain the
   target names/values and declared missing codes. Do not merge left/right
   knees or different visits, and do not duplicate rows across assessments.
   Demographics and identifiers are not diagnosis targets. Choose the official
   assessment release/reader before training and record that provenance.

4. Declare every target column in `tasks.json`. This is the **format**, using
   illustrative names; substitute actual release variables, legal values,
   missing codes, bounds and dictionary references:

   ```json
   {
     "tasks": [
       {"name": "native_binary_finding", "kind": "binary",
        "missing_values": ["-1"], "provenance": "Actual release and dictionary field"},
       {"name": "native_grade", "kind": "categorical",
        "classes": ["0", "1", "2", "3"], "missing_values": ["-1"],
        "provenance": "Actual release and dictionary field"},
       {"name": "native_score", "kind": "regression",
        "minimum": 0, "maximum": 100, "missing_values": ["-1"],
        "provenance": "Actual release and dictionary field"}
     ]
   }
   ```

   Bounds/classes come from the release dictionary, not training/validation
   statistics. Every declared task needs at least one observed training label.
   Unknown label values, undeclared target columns, ambiguous joins and
   participants crossing partitions fail preparation. The per-task coverage
   report lists observation counts and distinct values, including constant tasks.

The code cannot create OAI diagnoses from unlabelled images. If you have only
OAI images, obtain the associated assessment tables before calling this a run
on all datasets **and labels**. The three OAI files are required, not optional.

## Run on the A4500

After updating the main checkout without discarding local notebook edits:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
python -m pip install -e '.[b58,test]'
python -m pytest -q developments/tests/test_b58_full_data.py

export B58_DATA_ROOT="$PWD/rsna-knee-abnormality-detection"
export B58_MRNET_ROOT="/home/talafha/data/MRNet"
export B58_FASTMRI_ROOT="$PWD/data/external/fastmri"
export B58_OAI_ROOT="$PWD/data/external/oai"
export B58_FASTMRI_ANNOTATIONS="$PWD/data/external/fastmri-plus/Annotations/knee.csv"
export B58_FASTMRI_REVIEWED="$PWD/data/external/fastmri-plus/Annotations/knee_file_list.csv"
export B58_OAI_SERIES="$PWD/data/external/oai_labels/series.csv"
export B58_OAI_LABELS="$PWD/data/external/oai_labels/labels.csv"
export B58_OAI_TASKS="$PWD/data/external/oai_labels/tasks.json"
```

Adjust external paths to their real locations. Set `B58_LABELS_ROOT`,
`B58_SERIES_POLICY` and `B58_SELECTION_ROOT` to your existing RSNA artifacts if
they differ from the original runbook defaults. With all inputs ready:

```bash
bash developments/scripts/run_b58_full_data.sh prepare && \
bash developments/scripts/run_b58_full_data.sh run --device cuda --num-workers 2
```

`prepare` hashes labels and every raw volume, validates pixels, prints complete
source/task counts, and freezes `protocol.json`/`protocol.sha256`. A retry uses
verified per-volume audit metadata; it never changes frozen membership.
Only metadata is cached. Training rereads and verifies original files, so disk
speed matters. The run may be much longer than v1; exact steps are printed
before training rather than an unmeasured hours estimate.

`run` launches separate processes to free GPU memory between stages. It checks
real forward/backward passes from all four sources and the largest labelled
case in each, then checks the full RSNA model before fine-tuning. No optimizer
step occurs during preflight. The native-label stage defaults to one case per
step, encoder chunk size two, and checkpointed activations. Its images stream
in-process; `--num-workers` controls only final RSNA loading.

Repeat the same `run` command after an interruption. Adaptation saves model,
optimizer, scaler, RNG, schedule position and source exposure every 100 steps
and at every epoch boundary. RSNA fine-tuning retains the existing epoch-boundary
recovery. Repeating a completed stage verifies its checkpoint and returns it.
Do not edit source/config/labels or replace source images while a run is active.

To run stages individually:

```bash
bash developments/scripts/run_b58_full_data.sh preflight
bash developments/scripts/run_b58_full_data.sh pretrain
bash developments/scripts/run_b58_full_data.sh native-preflight
bash developments/scripts/run_b58_full_data.sh supervised
bash developments/scripts/run_b58_full_data.sh finetune --num-workers 2
```

## Evidence and interpretation

Outputs include the frozen manifest and label coverage, stage histories,
source exposure counts, final adapted encoder, native heads, the fixed-final
RSNA checkpoint, and validation/expert prediction exports. Compare only after
both complete, using the original B57 package-run export:

```bash
bash developments/scripts/run_b58_full_data.sh compare \
  --reference-root /path/to/completed/b57_package_run
```

The new standalone notebook uses another export format; do not point this
comparator at it. The comparator verifies the original label/split identities,
reports per-target and macro AUC, and uses the existing paired bootstrap. The
development support threshold remains `+0.010` with a positive lower confidence
bound; the reused development split is not an independent final test.

This revision changes data volume, external labels, encoder adaptation and
training duration together. Any gain supports the combined recipe; it cannot
identify which change caused it. No improvement, automatic promotion, Kaggle
submission, or adaptation of the running B57 checkpoint is claimed.
