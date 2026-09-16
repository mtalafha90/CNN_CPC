# Standalone DINOv2 knee MRI notebook

Open [`dinov2_knee_mri_local.ipynb`](../notebook/dinov2_knee_mri_local.ipynb)
for the complete local DINOv2 workflow on one RTX 5090. Every pipeline function
and class is defined in notebook cells: DICOM/metadata handling, supervision,
scanner separation, preprocessing, the dataset, study model, loss, optimizer,
preflight, training, recovery, evaluation, inference loading and plotting.

**Copying this one notebook is sufficient for the code.** It does not import
`rsna_knee`, alter `sys.path`, run external scripts, load Python helpers, or
require an editable project installation. Standard Python libraries are still
required; the DINOv2 image backbone comes from `timm==1.0.20` and the authors'
public weights. All knee-specific architecture code is in the notebook.

The established recipe is preserved: public ViT-S/14, full encoder fine-tuning,
32 three-slice inputs per series, 90% crop, aspect-preserving geometry at about
448² pixels, one series transformer, twelve finding queries, two studies per
optimizer update and twelve fixed epochs. Full data means every permitted
training study; scanner validation and expert studies remain held out.

## Open the updated notebook on the 5090

Use the primary project directory. Updating fetches the notebook; the running
notebook does not depend on the checkout. Preserve local edits if Git reports
a conflict.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git status --short
git pull --ff-only origin main
conda activate dinov2-local
python -m jupyterlab notebook/dinov2_knee_mri_local.ipynb --ServerApp.ip=127.0.0.1
```

If Jupyter is already open, close the old notebook tab, reopen the file from
the file browser, and restart its kernel. Otherwise an open editor may still
contain the earlier package-based notebook. Keep any previously executed copy
under a different filename if you need its output record.

If the notebook environment does not yet exist, create it once by cloning the
environment whose PyTorch works on the 5090. This preserves its working GPU
build. Activate an existing environment instead of recreating it.

```bash
conda create --name dinov2-local --clone rsna-knee -y
conda activate dinov2-local
python -m pip install "timm==1.0.20" "numpy>=1.26" "pandas>=2" \
  "scikit-learn>=1.3" "matplotlib>=3.8" "pydicom>=3" "python-gdcm>=3.0.10" \
  "jupyterlab>=4,<5" "tornado==6.5.8" "ipykernel>=6"
python -m ipykernel install --user --name dinov2-local --display-name 'Knee MRI · DINOv2'
```

No `pip install -e .` is needed. Keep the working Torch/torchvision pair. The
notebook performs CUDA arithmetic and a real model backward preflight before
training. The first public-weight preparation may need internet access; later
runs use the verified local initialization. Registering a kernel from its own
environment follows the standard
[IPython workflow](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Inputs and independent execution

The first cell defaults to the existing primary directory as `WORK_ROOT`, the
full original dataset, and a dedicated output folder:

```text
/media/talafha/Disk_1/CNN_CPC/runs/dinov2_knee_mri_standalone
```

`WORK_ROOT` is just a data/output directory. No source directory is required.
You can put the notebook anywhere and set the paths explicitly. The required
inputs are:

| Setting | Required files |
|---|---|
| `DATA_ROOT` | Original `train.csv`, `train_series.csv`, and DICOM series |
| `LABELS_ROOT` | Original `training_targets.csv`, `policy.json`, `audit.json` |
| `SCANNER_SPLIT_ROOT` | Existing selection-split JSON, per-study CSV and SHA256 file |
| `SERIES_POLICY` | Existing label-free all-series policy JSON |
| `RUN_ROOT` | A dedicated empty folder for a new run, or this notebook's existing run |

Leave the three optional input settings as `None` to discover them under
`WORK_ROOT/runs`. Discovery first reads existing frozen protocol JSON files
for verified input locations. These are data-only manifests; no old Python
implementation or model is imported. If there are multiple candidates, set
the explicit paths. Do not rebuild the split or replace the teacher labels.

Preparation reconstructs the original soft labels and scanner grouping from
raw CSV/JSON files and DICOM headers, so an older prepared protocol is not
required. Original artifact filenames/column names are accepted without
renaming those files. The default population is 3,603 training, 548 validation,
198 profile-overlap exclusions and 58 expert diagnostic studies.

Supported DICOM layouts are `train_series/<study>/<series>`,
`train_images/<study>/<series>`, or `<study>/<series>` under `DATA_ROOT`.
Supported suffixes include `.dcm`, `.dicom`, `.ima`, and no suffix. The inherited
loader can skip individual decode failures; missing or wholly unreadable
series raise errors. This is not a complete file-by-file pixel audit.

## Run and resume

1. Select **Knee MRI · DINOv2** and review the paths in the first cell.
2. Run the definition cells in order. They contain all executable pipeline code.
3. In **Run the workflow**, prepare inputs and public weights, inspect the
   population and actual slice preview, and run the model preflight.
4. Start the training cell when the GPU is available. Leave Jupyter running.
5. After training, run the learning-curve and results cell. Save your executed
   notebook copy as the run record.

The notebook uses `num_workers=0` so its dataset class can run directly in
Jupyter without exporting a module for spawned workers. This can reduce
loading throughput relative to multiprocessing; it leaves the training
recipe unchanged. Do not select spawned workers for notebook-local classes.

Training runs in the kernel, and ordinary interruption cleans up model/optimizer
references. Recovery is at completed epochs; a partial epoch is repeated.
After restarting the kernel, keep the same code, settings, environment and
inputs, rerun the cells, and start training again. The checkpoint restores
model, optimizer, scheduler, scaler and random state. A completed run verifies
its final artifacts and returns without additional updates.

This notebook has a separate checkpoint format and default output folder.
It does not resume the earlier package-based runner's checkpoints. Existing
training runs and their source modules are preserved. Model state keys and
numerical behavior are checked against the established DINOv2 implementation,
but this is not a claim that a new GPU run will be bitwise identical.

The resume contract fingerprints the executed function definitions, recipe,
input files, pretrained tensors, library versions and precision. Copying the
notebook to a different filename is supported. Moving an existing run and its
inputs to different paths or changing software can invalidate its resume
contract; independent new runs may use any correct paths.

## Outputs

| Folder or file under `RUN_ROOT` | Contents |
|---|---|
| `notebook_run.json` | Executed implementation fingerprint and recipe |
| `protocol/` | Input hashes, frozen labels, study lists and series index |
| `public_init/` | Hash-verified public image-encoder initialization |
| `preflight.json` | Actual backward checks, zero updates and peak GPU allocation |
| `model/` | Recovery/final checkpoints, history and held-out predictions |
| `reports/` | Metrics, per-finding AUC/coverage and per-study probabilities |
| `figures/` | Architecture, slices, sigmoid, history and AUC as PNG/SVG |

The saved model can be reconstructed with the notebook's own
`load_trained_model` function. All outputs are for the fixed final epoch.
The reused scanner validation surface is a development measure, and expert
scores are diagnostic only. This workflow does not select the best epoch,
compare paired candidate/control models or generate competition submissions.

## Repair a blank JupyterLab page or static-file HTTP 500

The following traceback was reproduced with JupyterLab 4.6.3, Jupyter Server
2.21.0 and Tornado 6.5.9 on 2026-09-15:

```text
AttributeError: 'FileFindHandler' object has no attribute 'allowed_symlink_directory'
```

Tornado 6.5.9 added a static-file symlink check, but Jupyter's custom handler
does not initialise its new attribute. The failure occurs while serving the
JavaScript bundle and favicon, before the notebook loads. The earlier notebook
kernel checks did not exercise this browser-asset path. See the
[Tornado release change](https://github.com/tornadoweb/tornado/blob/v6.5.9/docs/releases/v6.5.9.rst)
and [Jupyter's handler](https://github.com/jupyter-server/jupyter_server/blob/v2.21.0/jupyter_server/base/handlers.py).

The notebook dependency extra now pins Tornado to **6.5.8**, which passes the
same static-asset checks. For an already installed environment, stop the failed
Jupyter server with **Ctrl+C** and confirm with `y` if prompted. In the same
terminal, keeping the environment that launched the server active, run:

```bash
cd /media/talafha/Disk_1/CNN_CPC
python -m pip install --no-deps 'tornado==6.5.8'
python -c "import sys, tornado; print(sys.executable); print('Tornado:', tornado.version)"
python -m jupyterlab notebook/dinov2_knee_mri_local.ipynb --ServerApp.ip=127.0.0.1
```

This changes only Tornado. It does not reinstall PyTorch, CUDA, or the training
packages. Use the new URL printed by the restarted server, then reload the
browser with **Ctrl+Shift+R**. The message about skipped, non-installed language
servers is informational and is not the cause of this failure.

This temporary pin is for the documented loopback-only local server; it
predates the symlink hardening in Tornado 6.5.9. Retire the pin once a compatible
upstream combination passes the static-asset tests. Authentication and XSRF
settings remain at their normal defaults.

## Maintainer checks

The notebook builder extracts selected existing algorithms at **build time**
and inserts their complete definitions into visible notebook cells. It adds
standalone orchestration in the builder. Notebook execution never calls the
builder or reads repository source/config files. Edit the builder and rebuild:

```bash
python notebook/build_dinov2_local_notebook.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg python -m pytest -q \
  notebook/test_dinov2_local_notebook.py \
  notebook/test_local_dinov2_adapter.py \
  notebook/test_jupyter_static_assets.py \
  developments/tests/test_b57_clean_comparison.py
```

Tests check exact preprocessing agreement, real DINOv2 state/logit/gradient
agreement, label/scanner metadata boundaries, interrupted versus uninterrupted
training, checkpoint round trips and tamper rejection. Every notebook cell
also runs from an isolated directory with repository imports blocked and
explicitly synthetic CPU fixtures. CI adds execution in a real Jupyter kernel.
The earlier adapter keeps its compatibility tests. No GPU/full-data training
result is implied by these checks; use the notebook's actual 5090 preflight.
