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

### 2. Your trained model

Upload the checkpoint on its own as a second private dataset:

```text
runs/control/train/all-script/model.pt
```

Call it something like `cnn-cpc-model`.

The file is large, so this upload takes a while. Do it once and reuse it.

## Making the notebook

1. On the competition page, click **New Notebook**.
2. In the **Add Input** panel, attach three things:
   - the competition data
   - your `cnn-cpc-code` dataset
   - your `cnn-cpc-model` dataset
3. Turn the **GPU on** in the settings panel.
4. Turn the **internet off**. Code competitions require this, and your model
   does not need it.
5. Copy the blocks from [`submission_notebook.py`](submission_notebook.py) into
   cells, in order.
6. Run cell 1 and read what it prints. Fix the two paths in cell 2 to match.
7. **Save & Run All**.
8. When it finishes, open the notebook's Output tab and click **Submit**.

## Which model to submit

Use the **report-aligned** one. Not DINOv3.

DINOv3 fetches its weights over the internet the first time it is used, and the
notebook runs with the internet switched off, so it would fail. Cell 4 checks
for this and stops early with a clear message rather than wasting the run.

## If something goes wrong

**Cell 1 takes forever** — you have an old copy of the notebook. Cell 1 must
only list the top level of each folder. The competition data holds hundreds of
thousands of scan files, and listing them all takes many minutes for nothing.

**"could not find the repository"** — the folder name in cell 2 does not match
what cell 1 printed. Fix the name.

**"no attached folder contains test.csv"** — the competition data is not
attached. Add it in the Add Input panel.

**It runs out of time** — the notebook has a time limit. Your local run takes a
few minutes for 3 knees, but the hidden set is much larger. If it times out,
lower `b7_eval_batch_size` in the config, or ask for help before rerunning,
since each attempt costs a slot.

**Every column is constant** — cell 6 warns about this. It means the model is
giving every knee the same answer, which is what the DINOv3 model did. Do not
submit that.

## What a submission costs you

Nothing on your own machine. Kaggle runs it on theirs. The only limit is how
many submissions per day the competition allows.
