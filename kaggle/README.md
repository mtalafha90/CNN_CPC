# Submitting to the competition

This is a **code competition**. You do not upload a finished answer file.
You upload a notebook, and Kaggle runs it against test data you never see.

That is why the answer file you made locally only had 3 knees in it. Those
three are a practice set. The real ones are hidden.

## What you need to upload first

Two Kaggle datasets. Make them once, then reuse them for every submission.

### 1. Your code

Zip the repository and upload it as a private Kaggle dataset.

```bash
cd /media/talafha/Disk_1/CNN_CPC
zip -r cnn_cpc_code.zip \
  model/ data/ training/ validation/ testing/ config/ \
  developments/src/ \
  -x '*__pycache__*' '*.pyc'
```

Call the dataset something like `cnn-cpc-code`.

You do not need `runs/`, `developments/docs/`, `developments/tests/` or the
dataset itself. Keeping it small makes the notebook start faster.

### 2. Your trained models

Put every checkpoint you want to submit into one folder first, each with its
own name. The notebook averages every `.pt` it finds, so two files with the
same name means one silently replaces the other and you submit a single model
by accident.

```bash
mkdir -p ~/kaggle_models
cp runs/control/train/all-script/model.pt   ~/kaggle_models/model_frozen.pt
cp runs/finetune/train/all-script/model.pt  ~/kaggle_models/model_finetuned.pt
ls -lh ~/kaggle_models
```

Upload that folder as a private dataset. The files are large, so this takes a
while; do it once and reuse it.

**Adding another model later.** Open the dataset page, click the **⋮** button,
choose **New Version**, and drag the new `.pt` in beside the existing ones.
Then go back to the notebook, find the dataset in the **Input** panel, and
click **Check for updates** — a notebook keeps using the version it was
attached to until you tell it otherwise, so skipping this leaves the new model
invisible.

## Making the notebook

1. On the competition page, click **New Notebook**.
2. In the **Add Input** panel, attach three things:
   - the competition data
   - your code dataset
   - your model dataset
3. Turn the **GPU on** in the settings panel.
4. Turn the **internet off**. Code competitions require this, and your model
   does not need it.
5. Copy the five blocks from [`submission_notebook.py`](submission_notebook.py)
   into cells, in order.
6. Run cells 1 to 3. They finish in seconds and print where everything was
   found. Nothing needs editing -- the cells search rather than assume.
7. **Save & Run All**. Cell 4 is the long one; that is where the scans are read.
8. When it finishes, open the notebook's Output tab and click **Submit**.

Cells 1, 2, 3 and 5 take seconds. Only cell 4 is slow, and it should be.

## Choosing which models to submit

Cell 2 has one line you edit, and it is the only one:

```python
SUBMIT = ["model_finetuned.pt"]
```

Name the checkpoints this run should use. Anything else in the dataset is
listed as `not submitting:` and ignored. To average two models, name both.

The notebook used to take every `.pt` it found. That sounds convenient and is
a trap: a model dataset gathers checkpoints as a competition goes on, so
uploading one for next week silently turns this week's single-model run into
an ensemble. Naming them keeps the run under your control.

Cell 3 then prints, for each named checkpoint, whether its encoder was
fine-tuned. It works this out from the fingerprints taken before and after
training, not from the filename -- a filename is a label somebody typed, and
it can be wrong.

## Read the output before you spend a submission

A submission slot is the one thing here you cannot get back, so check three
things in the run you are about to submit:

- **Cell 3** prints `submitting N model(s)`, and one block per model saying
  whether its encoder was fine-tuned. Check that it matches the run you meant.
- **Cell 4** prints `[ensemble] averaging N models` when there is more than one.
- The manifest cell 4 prints has `"ensemble_size"`. If the key is missing
  entirely, the notebook is running an old copy of the code dataset — upload a
  new version of it and click **Check for updates**.

`"test_studies": 3` is normal here and is not a fault. Kaggle gives you a
three-knee practice set while you build the notebook and swaps in the hidden
set only when you press Submit.

## Which model to submit

Use the **report-aligned** one. Not DINOv3.

DINOv3 fetches its weights over the internet the first time it is used, and the
notebook runs with the internet switched off, so it would fail. Cell 4 checks
for this and stops early with a clear message rather than wasting the run.

## If something goes wrong

**Cell 1 takes forever** — you have an old copy of the notebook. Cell 1 must
only list the top level of each folder. The competition data holds hundreds of
thousands of scan files, and listing them all takes many minutes for nothing.

**"repository not found"** — cell 2 prints what the dataset actually contains
when this happens. Kaggle sometimes wraps an uploaded zip in an extra folder,
so the code can end up one or two levels deeper than expected. Cell 2 searches
for it rather than guessing, so this usually means the wrong dataset name in
`CODE_DATASET`, or the zip did not upload fully.

**"no attached folder contains test.csv"** — the competition data is not
attached. Add it in the Add Input panel.

**It runs out of time** — the notebook has a time limit. Your local run takes a
few minutes for 3 knees, but the hidden set is much larger. If it times out,
lower `b7_eval_batch_size` in the config, or ask for help before rerunning,
since each attempt costs a slot.

**Every column is constant** — cell 5 warns about this. It means the model is
giving every knee the same answer, which is what the DINOv3 model did. Do not
submit that.

**Only one model is listed, but you attached two** — either both files are
called `model.pt`, so one overwrote the other, or the notebook is still on the
old version of the dataset. Rename the files and click **Check for updates**.

## What a submission costs you

Nothing on your own machine. Kaggle runs it on theirs. The only limit is how
many submissions per day the competition allows.
