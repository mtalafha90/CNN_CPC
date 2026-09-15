# Local DINOv2 knee MRI notebook

Open [`dinov2_knee_mri_local.ipynb`](../notebook/dinov2_knee_mri_local.ipynb)
for the explained full-data DINOv2 workflow on one RTX 5090. It covers data
boundaries, real MRI slice previews, the architecture, sigmoid and loss, a
gradient preflight, training/recovery, and final AUC and probability exports.

The notebook uses the existing DINOv2 training recipe: public ViT-S/14 encoder,
all layers trainable, 32 three-slice inputs per series, aspect-preserving
geometry, one series transformer, twelve finding queries, two studies per
optimizer step and twelve fixed epochs. The original label and scanner split
artifacts are required. Full data means every allowed training study each
epoch; scanner validation and expert diagnostic studies remain held out.

## Setup on the 5090 machine

Finish the current GPU training job before updating its checkout or starting
another training process. Work inside the primary project directory. The
commands below never create a sibling project checkout.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git status --short
git pull --ff-only origin main
test -f notebook/dinov2_knee_mri_local.ipynb
```

If Git stops because of local edits, preserve and resolve those edits before
continuing; do not reset or remove existing training artifacts.

Create a notebook environment by cloning the environment whose PyTorch already
works on the 5090. This preserves its GPU build and keeps notebook dependencies
separate. Do this creation step once. If `dinov2-local` already exists, activate
it instead of recreating it.

```bash
conda create --name dinov2-local --clone rsna-knee -y
conda activate dinov2-local
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name dinov2-local --display-name 'Knee MRI · DINOv2'
python -m jupyterlab notebook/dinov2_knee_mri_local.ipynb --ServerApp.ip=127.0.0.1
```

The editable install has no upgrade flag. Keep the working Torch/torchvision
pair; the notebook performs an actual CUDA forward/backward check and a real
model preflight. Registering a kernel from its own environment is the standard
[IPython kernel installation workflow](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

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

## In Jupyter

1. Select **Knee MRI · DINOv2** as the kernel.
2. Review the first settings cell. The defaults point to the primary checkout
   and its full `rsna-knee-abnormality-detection` dataset.
3. Keep the default output folder `runs/dinov2_knee_mri_local` for a new run.
   It is separate from your current training artifacts. A nonempty foreign run
   folder is refused.
4. Leave the label, scanner-split and series-policy overrides as `None` when
   they remain in their established locations. Existing protocol paths are
   reused when available; otherwise the original artifacts are discovered.
   Missing or ambiguous artifacts require their exact paths, not a new split.
5. Run the cells in order. Review the population table and real slice preview,
   wait for the model preflight to pass, then run the training cell.
6. After training finishes, run the history and evaluation cells and save the
   executed notebook as your local run record.

The source notebook has no experiment-number labels and ships with empty
outputs. The adapter retains the original machine-readable protocol identities
and checkpoint keys internally for compatibility; it does not rewrite model
provenance. Only the DINOv2 encoder is initialized and trained by this workflow.

## Recovery and outputs

Normal **Kernel → Interrupt** signals the trainer and data-worker process
group. Recovery is at the last completed epoch. An unfinished epoch is repeated.
After restarting Jupyter, use the same settings, code and environment, rerun
setup/preparation/preflight, then run the training cell again. A completed run
is checked and returned without training again. Leave the Jupyter server and
kernel running during a live run.

`NUM_WORKERS = 2` is a conservative default. Set it to zero for loading-error
diagnosis or lower CPU memory pressure. Prefetch remains one batch per worker.
Worker changes are allowed on resume; model, precision, input and software
contract changes are checked and rejected. The notebook does not expose a new
slice-count, geometry or epoch sweep.

All outputs are inside `CNN_CPC/runs/dinov2_knee_mri_local/`:

| Folder or file | Contents |
|---|---|
| `notebook_session.json` | Input paths and notebook adapter fingerprints |
| `protocol/` | Frozen input hashes, study groups, labels and model settings |
| `public_init/` | Hash-verified public DINOv2 initialization |
| `dinov2_slice_candidate/` | Recovery/final checkpoints, history and prediction artifacts |
| `reports/` | Final metrics, per-finding coverage/AUC and per-study probabilities |
| `figures/` | Architecture, slices, sigmoid, learning curves and AUC as PNG/SVG |
| `logs/` | Verbatim stage logs and full preflight details |

Evaluation is for the fixed final epoch on the existing development and expert
diagnostic surfaces. The notebook does not select the best epoch, calculate a
paired control comparison, or create a competition submission.

## Maintainer checks

The notebook is generated deterministically; edit its builder and rebuild it.
The local helper is a subpackage so existing flat-module source digests remain
unchanged. Its own adapter files are fingerprinted separately for resume.

```bash
python notebook/build_dinov2_local_notebook.py
python -m pip install -e '.[test,notebook]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg python -m pytest -q \
  notebook/test_dinov2_local_notebook.py \
  notebook/test_jupyter_static_assets.py \
  developments/tests/test_b57_clean_comparison.py
```

Automated notebook execution uses explicitly synthetic CPU fixtures. Production
data/model preparation, train/resume delegation, process interruption and report
hash/label checks have separate tests; the existing model regression suite
checks real DINOv2 gradients and checkpoint round trips. CI also installs the
actual notebook web dependencies and validates the shipped JavaScript and
favicon through Jupyter's static-file handler. These checks are not
a substitute for the included real-data preflight on the 5090.
