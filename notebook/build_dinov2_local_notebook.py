"""Build the clean, explanatory local DINOv2 notebook deterministically."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

CELLS = []


def markdown(source):
    CELLS.append(("markdown", dedent(source).strip()))


def code(source):
    CELLS.append(("code", dedent(source).strip()))


markdown(r"""
# Knee MRI diagnosis with DINOv2

**Local training · full original dataset · one NVIDIA RTX 5090**

This notebook trains a multi-label model that reads the MRI series in a knee
study and estimates twelve finding probabilities. It uses a publicly pretrained
DINOv2 image encoder, a small transformer to combine slices, and attention to
combine series. All encoder layers and the new diagnosis heads are trained.

The notebook runs the repository's established DINOv2 implementation. Model
code stays in the Python package; the cells explain the pipeline, check your
inputs, launch training, and display the actual results. Figures of the
architecture and sigmoid are explanatory. MRI previews and metric plots come
from your data and saved run.

**Full data** means the full local dataset and every allowed training study
once per epoch. Scanner validation studies and the expert diagnostic set stay
outside training. This is the same existing split and recipe, with no subset
sampling and no external MRI datasets.

Run the cells in order. Start the GPU stages after your other GPU job has
finished. Keep the Jupyter server and kernel running during training.
""")

markdown(r"""
## 1. Local setup

Use the Python environment whose PyTorch installation already works on your
5090. A generic CUDA installation can detect the card yet fail during actual
computation, so a real forward/backward check follows below.

From the primary `CNN_CPC` checkout, the companion
[local setup guide](../docs/DINOV2_LOCAL_NOTEBOOK.md) installs the notebook
dependencies and registers the **Knee MRI · DINOv2** kernel. Select that kernel
in Jupyter. Do not install or upgrade packages halfway through training.

The default paths below match the local machine. Change `PROJECT_ROOT` and
`DATA_ROOT` if necessary. `RUN_ROOT` is a new folder inside the primary
project; it never automatically resumes another training job.

The original label export, scanner split and series policy are reused from
your existing local artifacts. Leave the optional paths as `None` for automatic
discovery. If an artifact was moved, set its exact existing path; the notebook
checks the frozen hashes and does not regenerate it.
""")

code("""
from pathlib import Path
import sys

PROJECT_ROOT = Path("/media/talafha/Disk_1/CNN_CPC")
DATA_ROOT = PROJECT_ROOT / "rsna-knee-abnormality-detection"
RUN_ROOT = PROJECT_ROOT / "runs/dinov2_knee_mri_local"

LABELS_ROOT = None
SCANNER_SPLIT_ROOT = None
SERIES_POLICY = None
NUM_WORKERS = 2

print("Notebook Python:", sys.executable)
print("Project:", PROJECT_ROOT)
print("Data:", DATA_ROOT)
print("Output:", RUN_ROOT)
""")

markdown(r"""
`NUM_WORKERS` controls CPU data-loading processes, not the model. Two is a
conservative starting point for full-resolution multi-series studies. Use zero
to diagnose loading errors or reduce host RAM usage. More workers help only
when CPU decoding or disk access is limiting the GPU.

The existing loader prefetches one batch per worker. Pinned memory and
persistent workers are disabled. These choices keep large variable-size MRI
bags manageable. The model uses two studies per optimizer update, processed
sequentially with weighted gradient accumulation, and encodes two three-slice
inputs at a time. Neither value means that the dataset is reduced to two studies.
""")

code("""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

source = (PROJECT_ROOT / "developments/src").resolve()
if not (source / "rsna_knee/local_dinov2").is_dir():
    raise FileNotFoundError("Update the primary checkout and check PROJECT_ROOT before continuing.")
sys.path.insert(0, str(source))
import rsna_knee
if Path(rsna_knee.__file__).resolve().parent != source / "rsna_knee":
    raise RuntimeError("This kernel imported a different checkout. Restart it and run these cells in order.")
from rsna_knee.local_dinov2 import LocalRun
from rsna_knee.local_dinov2 import plots

run = LocalRun(project_root=PROJECT_ROOT, data_root=DATA_ROOT, run_root=RUN_ROOT,
               labels_root=LABELS_ROOT, scanner_split_root=SCANNER_SPLIT_ROOT,
               series_policy=SERIES_POLICY, num_workers=NUM_WORKERS)
run.check_environment()
""")

markdown(r"""
The check above runs CUDA arithmetic and backpropagation in a separate process
and verifies the compressed DICOM decoders. That process exits afterward, so
Jupyter does not retain a CUDA model. Decoder availability is followed by real
image decoding in the preview and model preflight; a missing or wholly
unreadable series raises an error. These checks are not an audit of every file:
the existing series loader can skip individual files that fail decoding.

## 2. Freeze the data and public initialization

This step verifies the original CSV files, report-label export, series policy,
and existing scanner split. It saves their hashes, the study lists, and the
training recipe in this run's protocol. Every required MRI series directory
must exist and contain DICOM files. Missing inputs stop the run before training.

The encoder starts from the authors' **DINOv2 ViT-S/14** weights, pretrained by
self-supervised learning on the general-image LVD-142M collection. These are
not knee-specialist weights. The first use may download the public checkpoint;
cached weights are reused and their exact tensor fingerprint is verified.
The MRI diagnosis heads start fresh. No earlier competition-trained model is
loaded as a parent.

Repeated preparation verifies the same frozen inputs. It does not create a
new split, alter existing labels, or restart a completed run.
""")

code("""
run.prepare()
display(run.population())
display(run.label_coverage())
""")

markdown(r"""
The population table is computed from your actual frozen inputs. In the
established full-data split it contains 3,603 training studies, 548 scanner
validation studies, 198 additional studies excluded for scanner-profile
overlap, and 58 expert diagnostic studies.

Only the training rows produce gradients. Validation shares no scanner
profile with training. The expert set is used only for a final diagnostic
report. Scanner grouping is not proof of patient separation: patient identities
are unavailable in this contract.

The scanner validation surface has been inspected previously, so its AUC is a
development measure. It is not an untouched test score or a Kaggle score.
""")

markdown(r"""
## 3. From DICOM slices to model inputs

One study can contain several eligible sagittal, coronal and axial MRI series.
The model receives all series admitted by the frozen policy, rather than a
fixed count of series.

For each series, the existing loader:

1. Orders slices by physical position when orientation/position metadata is
   usable, with an instance-number fallback.
2. Normalizes volume intensity using its 1st and 99th percentiles, clipping
   values into the range 0–1.
3. Selects **32 slice centers** using the fixed dense sampling rule: 16 base
   centers plus 16 additional centers. Short series may repeat positions.
4. Builds a three-channel input from the previous, center and next slice.
   Boundary indices are clamped. Neighboring triplets overlap: this does not
   require 96 distinct slices.
5. Keeps the central 90% crop and resizes while preserving aspect ratio, to
   an area of approximately $448^2$ pixels. The existing stride padding is
   preserved; height and width vary between series.

The resulting tensor is **`[32, 3, H, W]` per series**. DINOv2 adds up to 13
reflect-padded pixels per edge dimension to make height and width divisible by
its patch size of 14, then applies its pretrained channel normalization.
These three channels contain adjacent grayscale slices, not RGB colors.

No random pixel augmentation or test-time augmentation is used in this recipe.
""")

code("""
example = run.training_example(study_index=0)
display(pd.DataFrame(example["geometry"])[["height", "width", "present"]]
        .rename_axis("Series index"))
fig = plots.slice_example(example, series_index=0)
plots.save_figure(fig, RUN_ROOT, "slice_inputs")
plt.show()
plt.close(fig)
del example
""")

markdown(r"""
## 4. Model architecture

The pretrained vision transformer splits each three-slice image into $14\times14$
patches and turns it into a **384-value feature vector**. Here, 384 is the
embedding width of the small DINOv2 encoder, not the image resolution and not
the number of slices. All its weights are fine-tuned on the allowed knee studies.

Within each series, a learned summary token joins the 32 slice vectors. One
transformer layer with six attention heads combines them into a single
384-value series vector. It has a 768-unit feed-forward block and dropout of 0.1.
Slice vectors are sorted by the supplied positions, but this block adds **no
slice-position or physical-spacing embedding**. Its context operation therefore
does not explicitly model distance or direction through the volume. The three
neighboring input slices provide local through-plane context.

Learned embeddings for scan plane, fluid sensitivity and fat suppression are
added to each series vector. Twelve learned finding queries then attend over
the available series. A residual connection and layer normalization produce
one feature vector per finding; twelve finding-specific linear heads output
the raw scores.
""")

code("""
fig = plots.architecture()
plots.save_figure(fig, RUN_ROOT, "architecture")
plt.show()
plt.close(fig)
""")

markdown(r"""
## 5. Logits, sigmoid, loss and AUC

For finding $j$, the classifier produces a logit
$z_j = \mathbf{w}_j^\mathsf{T}\mathbf{h}_j + b_j$. A logit is an unrestricted
real-valued score. During prediction, sigmoid converts it to

$$p_j = \sigma(z_j) = \frac{1}{1+e^{-z_j}}.$$

For example, logits −2, 0 and 2 become probabilities about 0.119, 0.500 and
0.881. Each finding is independent at the output: the twelve values do not
have to sum to one, because multiple abnormalities can coexist. A predicted
probability is not a guarantee of clinical calibration.

The twelve output columns are ACL, MCL, Medial Meniscus, Lateral Meniscus,
Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion and Fracture.
OA means osteoarthritis; PF means patellofemoral.

Training uses **binary cross-entropy on logits**, computed with a numerically
stable logits-based implementation. The original soft report targets and
confidence weights are retained. Unlabelled cells have weight zero and do not
contribute. Per-target balancing is computed from training confidence weights
so densely labelled findings do not simply dominate the loss. This DINOv2 arm
has no auxiliary local-head loss.

For reporting, ROC AUC measures ranking: how often a positive case is scored
above a negative case, with half credit for ties. The metric uses each target's
fixed active-label mask, converts the soft targets to binary labels at 0.5,
and computes **unweighted** AUC among those active rows. Macro AUC averages the
twelve target AUCs equally. An expert target with only one class has undefined
AUC; its coverage is reported and it is omitted from the expert macro.

Sigmoid preserves score order mathematically, so applying it does not by itself
improve AUC. Choosing a probability threshold such as 0.5 is also not part of
the AUC calculation.
""")

code("""
logits = np.array([-2.0, 0.0, 2.0])
display(pd.DataFrame({"Example logit": logits, "Sigmoid probability": 1 / (1 + np.exp(-logits))}))
fig = plots.sigmoid()
plots.save_figure(fig, RUN_ROOT, "sigmoid")
plt.show()
plt.close(fig)
""")

markdown(r"""
## 6. Fixed training recipe

The table below is read from the verified configuration, not a second copy
of model settings in this notebook. Training lasts **12 complete epochs**
with seed 2026. AdamW uses an initial learning rate of $10^{-5}$ for the
pretrained encoder and $10^{-4}$ for the new heads, with weight decay $10^{-4}$.
A cosine schedule reduces both rates toward 1% of their initial values.
The gradient norm is clipped at 1.0.

Automatic mixed precision uses the existing runtime policy (normally bfloat16
on a supported 5090), while gradient checkpointing recomputes encoder
activations during backward to reduce GPU memory use. The preflight records
the actual precision and software versions; changing those during resume is
rejected by the saved training contract.

Every epoch trains on every allowed training study exactly once, followed by
scanner validation. The **final epoch is the endpoint**. The validation curve
is shown for diagnosis; this notebook does not choose its highest-scoring
epoch, stop early, tune the split, or blend another model.
""")

code("""
display(run.recipe())
""")

markdown(r"""
## 7. Real-data model preflight

This builds the actual model, decodes two training studies with many eligible
series, and performs a forward/backward pass. It checks finite, nonzero
gradients in both the pretrained encoder and the new heads, including the
first and last encoder parameters. **No optimizer step is taken.**

It prints peak allocated CUDA memory for this check. That is not a bound for
every study: MRI series have different shapes and counts. The disposable model
exits with its process after the check. Successful CUDA arithmetic alone does
not replace this full model check.
""")

code("""
run.preflight()
""")

markdown(r"""
## 8. Train or resume

The next cell starts training. It streams progress and saves the full raw log
in `RUN_ROOT/logs`. A progress line is emitted every 100 training batches and
at the end of each epoch; the first line may take time while images load and
the GPU starts work. Final validation and expert prediction export also take
time after the last epoch.

An atomic recovery checkpoint is saved after each completed epoch, including
model weights, optimizer, scheduler, mixed-precision scaler, random states and
history. After an interruption, rerun the setup, preparation and preflight
cells with the same paths, code and environment, then rerun this cell. Training
continues from the last completed epoch; a partly completed epoch is repeated.
If interrupted before the first completed epoch, training starts from its
original initialization. A completed run is verified and left complete.

Use **Kernel → Interrupt** to stop normally. The launcher signals the trainer
and its data-worker process group. Avoid shutting down the kernel or terminal
as a stopping method. Do not edit source/config files or update the checkout
while a training run is in progress. Do not launch a second copy on the same GPU.
""")

code("""
run.train()
""")

markdown(r"""
## 9. Learning curves from saved epochs

These plots read the actual saved epoch history. Training loss and validation
AUC have different meanings and separate axes: a lower loss does not guarantee
a higher AUC. The displayed AUC curve does not change the fixed final endpoint.

You can run this cell after at least one full epoch has been saved, including
after a normal interruption. Each point represents one completed epoch.
""")

code("""
history = run.history()
display(pd.DataFrame([{
    "Epoch": row["epoch"], "Training loss": row["train_loss"],
    "Scanner macro AUC": row["validation"]["macro_auc"],
    "Minutes": row["epoch_minutes"],
} for row in history]).set_index("Epoch"))
fig = plots.training_history(history)
plots.save_figure(fig, RUN_ROOT, "learning_curves")
plt.show()
plt.close(fig)
""")

markdown(r"""
## 10. Final evaluation and readable exports

This cell requires completed training. It verifies the final checkpoint and
prediction hashes, exact study identities, frozen targets, and confidence
masks before calculating metrics. It reports this model alone.

The scanner score is the primary development report. Expert results provide
a separate diagnostic and are never used for gradient updates or checkpoint
selection. A higher development AUC alone does not establish a higher Kaggle
score. The tables show positive, negative and unlabelled counts so an AUC is
not interpreted without its label coverage.

Readable metric JSON, per-finding CSV tables and per-study probability CSVs
are written under `RUN_ROOT/reports`. These are validation/diagnostic exports,
not a competition submission. PNG and SVG figures are saved under
`RUN_ROOT/figures` for later use.
""")

code("""
report, tables = run.results()
display(pd.DataFrame([{
    "Population": name, "Studies": score["studies"],
    "Macro AUC": score["macro_auc"], "Targets with AUC": score["targets_defined"],
} for name, score in report["scores"].items()]).set_index("Population"))
for name, table in tables.items():
    print(name.capitalize())
    display(table.rename_axis("Finding"))
fig = plots.target_auc(tables)
plots.save_figure(fig, RUN_ROOT, "finding_auc")
plt.show()
plt.close(fig)
""")

code("""
display(pd.DataFrame({"Artifact": run.artifacts().keys(),
                      "Path": [str(p) for p in run.artifacts().values()]}).set_index("Artifact"))
print("Keep this entire output folder and the original frozen inputs for reproducible recovery.")
""")

markdown(r"""
## Reading and troubleshooting

If a file is missing, correct its path or restore the original artifact; do not
replace the full CSV with a subset or rebuild the scanner split. If a hash or
resume contract fails, use the code, inputs and environment that created that
run. A different set of inputs belongs in a different output folder.

If decoding or CPU memory is the problem, interrupt normally, set
`NUM_WORKERS = 0`, and rerun the setup cells before resuming. Worker count is
an execution setting and does not change the epoch shuffle or allowed studies.
If CUDA runs out of memory, first stop other GPU workloads; reducing workers
usually addresses host memory, not model activation memory. Do not silently
change image geometry or the number of slices to make this recipe fit.

Keep the saved notebook with its actual outputs as your run record. The
repository template is intentionally distributed with empty outputs.

Primary references:

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Authors' DINOv2 repository and pretrained models](https://github.com/facebookresearch/dinov2)
- [PyTorch binary cross-entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Registering a Jupyter kernel from a Python environment](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
""")


def notebook():
    cells = []
    for i, (kind, source) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"local-dino-{i:02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Knee MRI · DINOv2", "language": "python", "name": "dinov2-local"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    out = Path(__file__).with_name("dinov2_knee_mri_local.ipynb")
    out.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
