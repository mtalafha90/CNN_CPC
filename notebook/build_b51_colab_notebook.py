"""Generate the B51-shaped Google Colab notebook.

Two things separate this notebook from the base one it inherits from.

**It learns from the reports.** The base notebook keeps a study only when
``train.csv`` already carries a label for it, and in this competition that is
true for just the 58 expert-gold studies. The other 4,349 studies have blank
label columns and a written report sitting in the same file. Feeding a label
export derived from those reports turns a 58-study toy into a real training set,
and moves the 58 gold studies out of training and into an honest test.

**It adapts the study hierarchy.** Every model from B37 to B49 froze the part of
the network that turns encoded slices into twelve answers, so roughly 98% of
each score came from weights that had not seen a gradient since B34. B50 tested
the alternative on the real data and it held: `+0.011221` macro AUC on 548
unseen-scanner studies, all twelve targets improved, and the effect was larger
on the base path than on the combined one, which is where a change to the
hierarchy should land. That question is settled, so this notebook trains the
adapted setting rather than re-running the comparison. The comparison is kept
behind a flag for anyone who wants to watch it happen.

The label export is read from Drive rather than produced here. B23 labels with an
openly downloadable checkpoint run locally and pins repo id, revision, dtype and
greedy decoding into the export, so a run is reproducible. Re-deriving labels
inside Colab would throw that provenance away and cost a model download before
any training started.

Everything else -- the Drive archive contract, DICOM decoding, the 448 geometry,
the dataset and the sparse-MIL head -- is inherited unchanged from
``build_notebook.py``.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path

BASE_BUILDER = Path(__file__).with_name("build_notebook.py")
BASE_NAMESPACE = runpy.run_path(str(BASE_BUILDER))
CELLS: list[tuple[str, str]] = list(BASE_NAMESPACE["CELLS"])


def replace_cell(index: int, kind: str, text: str) -> None:
    """Replace one inherited cell without modifying the base builder."""
    CELLS[index] = (kind, text.strip("\n"))


def append_cell(kind: str, text: str) -> None:
    """Add a cell to the end of the notebook."""
    CELLS.append((kind, text.strip("\n")))


def find_cell(marker: str) -> int:
    """Locate an inherited cell by its content, so edits survive reordering."""
    matches = [index for index, (_, text) in enumerate(CELLS) if marker in text]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def rewrite_cell(marker: str, kind: str, text: str) -> None:
    """Replace the one inherited cell that contains a marker."""
    CELLS[find_cell(marker)] = (kind, text.strip("\n"))


def retitle_cell(marker: str, old: str, new: str) -> None:
    """Renumber an inherited heading now that B51 sections come before it."""
    index = find_cell(marker)
    kind, text = CELLS[index]
    if old not in text:
        raise RuntimeError(f"heading {old!r} not found in cell {index}")
    CELLS[index] = (kind, text.replace(old, new))


def insert_cells(marker: str, new_cells: list) -> None:
    """Insert B51's definitions immediately before an inherited cell."""
    index = find_cell(marker)
    CELLS[index:index] = [(kind, text.strip("\n")) for kind, text in new_cells]


# --- what this notebook is -------------------------------------------------

replace_cell(
    1,
    "markdown",
    """
# B51 — learn from the reports, and let the study hierarchy adapt

This notebook changes two things about how the model is trained.

**1. The reports become labels.** `train.csv` has twelve label columns and a
`Report` column. Only the 58 expert-gold studies have the twelve columns filled
in. The other 4,349 have nothing in them — but they do have a written report.

The base notebook keeps only rows that already have a label, so it trains on the
58 gold studies and nothing else. Feeding it labels read out of the reports gives
it thousands of studies instead, and frees the 58 gold ones to be used as an
honest test the model never trained on.

**2. The study hierarchy is allowed to learn.** Every model from B37 to B49 froze
it, so about 98% of each score came from weights that had not been updated in a
long time. B50 tested letting it learn, on the real data:

| surface | frozen | adapted | difference |
|---|---|---|---|
| combined | 0.763117 | 0.774336 | **+0.011221** |
| base | 0.762566 | 0.774243 | +0.011676 |
| local | 0.743541 | 0.753820 | +0.010278 |

All twelve targets improved, and the effect was **larger on the base path than on
the combined one** — exactly where a change to the hierarchy should show up.

That question is answered, so this notebook simply trains the adapted setting.
""",
)

replace_cell(
    2,
    "markdown",
    """
## What this notebook is, and is not

**It is** a runnable version of B51's training recipe on whatever subset of the
data fits in your Drive: report-derived labels for training, expert-gold labels
for testing, and a study hierarchy that adapts at 0.05x the head's learning rate.

**It is not** the B51 protocol in `developments/`. That one trains on all 4,349
report-only studies, starts from the Phase-9 checkpoint rather than from random
weights, and uses a frozen scanner-grouped validation gate. It takes about eight
and a half hours on an RTX A4500. This notebook starts from scratch on a subset,
so its absolute numbers mean nothing and must not be compared with a leaderboard
score.

What transfers is the shape of the result: whether the model learns anything from
report labels, and whether the gold studies it never saw get ranked better.

### What you need in your Drive folder

Beside `train.csv` and `train_series.csv`, put the label export:

```text
training_targets.csv
```

That is the file `b23_llm_labels.py` writes (`b6_report_labels.py` writes the
same shape). It has one row per report-only study, and for each of the twelve
targets a probability, a `__confidence` and a `__state`. Section 11 checks the
file before anything else happens.
""",
)


DEFINITIONS: list = []


def define(kind: str, text: str) -> None:
    """Collect a B51 definition cell, inserted before the training section."""
    DEFINITIONS.append((kind, text))


# --- the reports as labels -------------------------------------------------

define(
    "markdown",
    """
## 11. Read the report labels

The export gives four states per target, and each state has a fixed probability
and a fixed confidence:

```text
state          probability   confidence
positive           0.97         0.90
negated            0.03         0.90
uncertain          0.50         0.25
conflict           0.50         0.20
unmentioned        0.50         0.00
```

The last row is the important one. When a report does not mention a finding, that
is **not** evidence the finding is absent — the radiologist simply did not write
about it. So an unmentioned cell is given confidence `0.00`, which means "ignore
this cell entirely".

This is easy to get wrong in a way that quietly ruins training. An unmentioned
cell is stored as probability `0.50`, not as a blank. If it were fed to an
ordinary loss it would look like a real label, and the model would be pushed
towards 0.50 on the majority of cells. The confidence column is what stops that,
so this notebook never reads a probability without its confidence.
""",
)

define(
    "code",
    '''
REPORT_LABELS_FILENAME = "training_targets.csv"


def report_label_columns() -> list[str]:
    """The exact columns b23_llm_labels.py and b6_report_labels.py write."""
    columns = ["StudyInstanceUID"]
    for target in TARGETS:
        columns.extend([target, f"{target}__confidence", f"{target}__state"])
    return columns


def load_report_labels(path: Path) -> pd.DataFrame:
    """Read the label export and refuse anything that is not the agreed shape."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No report labels at {path}.\\n"
            "Export them with b23_llm_labels.py and copy training_targets.csv "
            "into the same Drive folder as train.csv."
        )

    frame = pd.read_csv(path)
    missing = [name for name in report_label_columns() if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing[:6]}")

    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path.name} lists the same study more than once")

    for target in TARGETS:
        confidence = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce")
        if confidence.isna().any() or float(confidence.min()) < 0 or float(confidence.max()) > 1:
            raise ValueError(f"{target}__confidence must be a number between 0 and 1")
    return frame


def weak_targets_and_confidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the export into a target matrix and a confidence matrix.

    A cell with zero confidence is blanked to NaN as well as being given zero
    weight. Either alone would be enough; doing both means a later change to the
    loss cannot accidentally start training on report silence.
    """
    targets = np.full((len(frame), len(TARGETS)), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame), len(TARGETS)), dtype=np.float32)

    for index, target in enumerate(TARGETS):
        column = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float32)
        weight = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").to_numpy(np.float32)
        used = weight > 0
        targets[used, index] = column[used]
        confidence[used, index] = weight[used]
    return targets, confidence


def split_gold_and_report_only(train_table: pd.DataFrame) -> tuple:
    """A study is 'gold' exactly when train.csv already carries a label for it."""
    written = train_table[TARGETS].apply(pd.to_numeric, errors="coerce")
    is_gold = written.notna().any(axis=1)
    return (
        train_table.loc[is_gold].reset_index(drop=True),
        train_table.loc[~is_gold].reset_index(drop=True),
    )


def select_report_training_studies(
    train_table: pd.DataFrame, labels: pd.DataFrame, records
) -> tuple:
    """Choose what trains and what is held back, and refuse a gold leak.

    Kept apart from model building so it can be checked on its own. A mistake
    here would not crash: it would quietly put expert-gold studies into training
    and make every score the notebook prints look better than it is.
    """
    gold_frame, report_only = split_gold_and_report_only(train_table)

    leaked = sorted(set(labels["StudyInstanceUID"]) & set(gold_frame["StudyInstanceUID"]))
    if leaked:
        raise ValueError(
            f"the label export contains {len(leaked)} expert-gold studies "
            f"(for example {leaked[0]}); it must hold report-only studies only"
        )

    keep = labels["StudyInstanceUID"].isin(set(report_only["StudyInstanceUID"])) & labels[
        "StudyInstanceUID"
    ].isin(set(records))
    train_frame = labels.loc[keep].reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(
            "No study is in the export, in train.csv as report-only, and in your "
            "extracted DICOM subset at once. Check that the export covers the "
            "studies you actually downloaded."
        )

    gold_usable = gold_frame.loc[
        gold_frame["StudyInstanceUID"].isin(set(records))
    ].reset_index(drop=True)
    return train_frame, gold_usable


def describe_report_labels(confidence: np.ndarray) -> dict:
    """How much supervision the reports actually provide, per target."""
    used = confidence > 0
    return {
        "studies": int(confidence.shape[0]),
        "cells_total": int(used.size),
        "cells_used": int(used.sum()),
        "coverage": float(used.mean()),
        "per_target_cells": {
            target: int(used[:, index].sum()) for index, target in enumerate(TARGETS)
        },
    }
''',
)


# --- the loss --------------------------------------------------------------

define(
    "markdown",
    """
## 12. A loss that respects confidence

Two adjustments to ordinary cross entropy.

**Each cell is weighted by its confidence.** A confident `positive` (0.90) counts
much more than an `uncertain` (0.25), and an unmentioned cell (0.00) counts not
at all.

**Each target is balanced by how much supervision it has.** Some findings are
written about far more often than others. Without this, the targets the reports
happen to discuss most would dominate every gradient, and the rare ones would
barely train. The multiplier is `mean(mass) / mass`, where a target's mass is the
total confidence it received across the training studies.

This is the same rule the real pipeline uses, so the notebook teaches the
behaviour the `developments/` code actually has.
""",
)

define(
    "code",
    '''
def target_balance_multipliers(confidence: np.ndarray) -> np.ndarray:
    """Give every target the same total say, whatever the reports talked about."""
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim != 2 or confidence.shape[1] != len(TARGETS):
        raise ValueError(f"confidence must have shape [N,{len(TARGETS)}]")

    mass = confidence.sum(axis=0)
    if not (mass > 0).all():
        empty = [TARGETS[index] for index in np.flatnonzero(mass <= 0)]
        raise ValueError(f"the reports gave no usable supervision for: {empty}")
    return (float(mass.mean()) / mass).astype(np.float32)


def report_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy weighted by per-cell confidence and per-target balance."""
    if logits.shape != target.shape or logits.shape != confidence.shape:
        raise ValueError("logits, target and confidence must have the same shape")

    # A blank target is unusable whatever its confidence claims.
    known = torch.isfinite(target).float()
    effective = confidence.float() * known * multiplier.to(logits.device)[None, :]

    denominator = effective.sum()
    if float(denominator.detach().cpu()) <= 0:
        # No usable cell in this batch. Return a real zero that still has a
        # gradient path, so the training step stays well defined.
        return logits.sum() * 0.0

    safe_target = torch.nan_to_num(target, nan=0.0)
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), safe_target.float(), reduction="none"
    )
    return (cell * effective).sum() / denominator.clamp_min(1e-8)


class ReportSupervision:
    """Per-study confidence, looked up by study UID rather than by row number.

    The dataset filters and reindexes the frame it is given, so a confidence
    array addressed by position would silently drift out of step with the studies
    the loader actually yields. Addressing by UID cannot drift.
    """

    def __init__(self, confidence_by_study: dict, multiplier: np.ndarray) -> None:
        self.confidence_by_study = confidence_by_study
        self.multiplier = torch.tensor(multiplier, dtype=torch.float32, device=DEVICE)

    def batch(self, study_uids: list) -> torch.Tensor:
        """The confidence rows for one batch, in the batch's own order."""
        missing = [uid for uid in study_uids if uid not in self.confidence_by_study]
        if missing:
            raise KeyError(f"no confidence recorded for {missing[:3]}")
        rows = np.stack([self.confidence_by_study[uid] for uid in study_uids])
        return torch.tensor(rows, dtype=torch.float32, device=DEVICE)
''',
)


# --- the adapted hierarchy -------------------------------------------------

define(
    "markdown",
    """
## 13. Let the study hierarchy adapt

The model already separates the two paths this experiment is about:

```text
encoder             reads pixels
global_projection   ┐
global_classifier   ┘  the study hierarchy -> base logits
sparse_head            the local branch    -> local logits
fusion_gate            tanh(g), how much the local branch is trusted
```

`logits = base + tanh(g) * local`.

Freezing the hierarchy means the first pair stops learning, which is what every
model from B37 to B49 did. Adapting it means they learn, slowly — at 0.05x the
head's rate, the value B50 used.

Two details below are easy to get wrong, and both are mistakes the real
implementation had to avoid.

**Gradients are not training mode.** `requires_grad` and `train()`/`eval()` are
independent. Freezing must be done with `requires_grad_(False)`, never by
switching modules to eval, which would also change dropout and normalisation.

**A frozen parameter must not reach the optimiser.** If it did, weight decay would
still move it, and the "frozen" setting would not be frozen.
""",
)

define(
    "code",
    '''
HIERARCHY_PREFIXES = ("global_projection.", "global_classifier.")
HIERARCHY_LR_SCALE = 0.05  # the value B50 froze, inherited unchanged


def hierarchy_parameter_names(model: nn.Module) -> list[str]:
    """Name every parameter that belongs to the study hierarchy."""
    return [
        name
        for name, _ in model.named_parameters()
        if name.startswith(HIERARCHY_PREFIXES)
    ]


def set_hierarchy_trainable(model: nn.Module, trainable: bool) -> int:
    """Freeze or unfreeze the hierarchy, and return how many parameters moved.

    Uses requires_grad only. The module's train/eval mode is deliberately left
    alone: changing it would also change dropout and normalisation, and the two
    settings would then differ in more than the one thing being tested.
    """
    lookup = dict(model.named_parameters())
    total = 0
    for name in hierarchy_parameter_names(model):
        lookup[name].requires_grad_(bool(trainable))
        total += lookup[name].numel()
    return total


def build_parameter_groups(model: nn.Module, head_lr: float) -> list[dict]:
    """Head at full rate, hierarchy at a reduced one, frozen tensors excluded.

    A frozen parameter handed to the optimiser would still be moved by weight
    decay, so the frozen setting would not actually be frozen. Only tensors that
    require gradients are included.
    """
    hierarchy_names = set(hierarchy_parameter_names(model))
    head, hierarchy = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (hierarchy if name in hierarchy_names else head).append(parameter)

    groups = [{"params": head, "lr": float(head_lr), "name": "encoder_and_head"}]
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * HIERARCHY_LR_SCALE,
                "name": "study_hierarchy",
            }
        )
    return groups


def describe_trainable(model: nn.Module) -> dict:
    """What is actually learning, so a setting can be checked rather than assumed."""
    hierarchy_names = set(hierarchy_parameter_names(model))
    counts = {"hierarchy": 0, "everything_else": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = "hierarchy" if name in hierarchy_names else "everything_else"
        counts[key] += parameter.numel()
    return counts


def read_fusion_gate(model: nn.Module) -> np.ndarray:
    """tanh(g): how much of the local branch reaches the score, per target."""
    return torch.tanh(model.fusion_gate.detach()).cpu().numpy()
''',
)


# --- building the run ------------------------------------------------------

define(
    "markdown",
    """
## 14. Build the run

Who trains on what:

```text
report-only studies, labels from the export   ->  training
the 58 expert-gold studies                    ->  test only, never trained on
```

Keeping gold out of training is what makes the score at the end mean something.
It is also what the `developments/` protocol does, where the checkpoint records
`gold_labels_used: False`.

The builder below refuses an export that contains gold studies, because a leak
there would be invisible in the result and would quietly inflate every number
the notebook prints.
""",
)

define(
    "code",
    '''
def build_report_supervised_experiment(
    paths: DrivePaths,
    config: RunConfig = CONFIG,
    *,
    adapt_hierarchy: bool = True,
    labels_path: Path | None = None,
) -> tuple[Experiment, ReportSupervision]:
    """Train on report-labelled studies; hold the expert-gold studies back."""
    set_seed(config.seed)
    validate_dataset(paths)

    train_table = pd.read_csv(paths.train_csv)
    train_table["StudyInstanceUID"] = train_table["StudyInstanceUID"].astype(str)
    series_table = pd.read_csv(paths.series_csv)
    records = build_series_records(series_table, config)

    labels = load_report_labels(labels_path or paths.data_root / REPORT_LABELS_FILENAME)
    train_frame, gold_usable = select_report_training_studies(train_table, labels, records)

    targets, confidence = weak_targets_and_confidence(train_frame)

    # A report that mentions none of the twelve findings supervises nothing. Such
    # a study would cost a full DICOM decode per epoch and teach nothing, and the
    # inherited preflight refuses a batch with no usable label at all.
    usable = confidence.sum(axis=1) > 0
    if not usable.all():
        print(f"skipping {int((~usable).sum())} studies whose report mentions no finding")
        train_frame = train_frame.loc[usable].reset_index(drop=True)
        targets, confidence = targets[usable], confidence[usable]
    if train_frame.empty:
        raise ValueError("no report in the export mentions any of the twelve findings")

    for index, target in enumerate(TARGETS):
        train_frame[target] = targets[:, index]

    confidence_by_study = {
        uid: confidence[row] for row, uid in enumerate(train_frame["StudyInstanceUID"])
    }
    # Gold labels are real, so every known gold cell carries full confidence.
    for uid in gold_usable["StudyInstanceUID"]:
        confidence_by_study[uid] = np.ones(len(TARGETS), dtype=np.float32)

    train_dataset = KneeMRIDataset(
        train_frame, records, paths, config, split="train", include_targets=True
    )
    validation_dataset = (
        KneeMRIDataset(
            gold_usable, records, paths, config, split="train", include_targets=True
        )
        if not gold_usable.empty else None
    )

    # Balance is measured over the studies the loader will really yield.
    used_confidence = np.stack(
        [confidence_by_study[uid] for uid in train_dataset.study_uids]
    )
    supervision = ReportSupervision(
        confidence_by_study, target_balance_multipliers(used_confidence)
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
        "collate_fn": collate_studies,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        if validation_dataset is not None else None
    )

    model = HighResolutionSparseMIL(config).to(DEVICE)
    set_hierarchy_trainable(model, adapt_hierarchy)
    optimizer = torch.optim.AdamW(
        build_parameter_groups(model, config.learning_rate),
        weight_decay=config.weight_decay,
    )

    experiment = Experiment(
        paths=paths,
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda"),
        train_loader=train_loader,
        validation_loader=validation_loader,
        # Balance is handled per target by the multiplier, so this stays neutral.
        positive_weight=torch.ones(len(TARGETS), dtype=torch.float32, device=DEVICE),
    )

    summary = describe_report_labels(used_confidence)
    print(f"training studies (reports) : {len(train_dataset)}")
    print(f"test studies (expert gold) : {0 if validation_dataset is None else len(validation_dataset)}")
    print(f"report cells used          : {summary['cells_used']:,} of {summary['cells_total']:,} "
          f"({summary['coverage']:.1%})")
    print(f"adapt_hierarchy            : {adapt_hierarchy}")
    print(f"trainable                  : {describe_trainable(model)}")
    print(f"optimiser groups           : {[group['name'] for group in optimizer.param_groups]}")
    if validation_dataset is None:
        print("WARNING: no expert-gold study is in your subset, so nothing can be scored.")
    return experiment, supervision
''',
)


# --- training --------------------------------------------------------------

define(
    "markdown",
    """
## 15. Train

One pass over the report-labelled studies per epoch, then a score on the gold
studies the model has never seen.

The fusion gate is printed each epoch. On the real data that number moved before
any score did: when the base path improves, the model leans on the local
correction less, and `|tanh(g)|` falls.
""",
)

define(
    "code",
    '''
def run_report_epoch(
    experiment: Experiment,
    loader: DataLoader,
    supervision: ReportSupervision,
    training: bool,
) -> dict:
    """One pass, using each cell's confidence instead of treating all cells alike."""
    experiment.model.train(training)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for batch in loader:
        # Read the UIDs before move_batch drops them.
        confidence = supervision.batch(list(batch["study_uid"]))
        volumes, present, metadata, position, target = move_batch(batch)
        del batch

        if training:
            experiment.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), autocast_context():
            output = experiment.model(volumes, present, metadata, position)
            combined_loss = report_weighted_bce(
                output.logits, target, confidence, supervision.multiplier
            )
            local_loss = report_weighted_bce(
                output.local_logits, target, confidence, supervision.multiplier
            )
            loss = combined_loss + experiment.config.local_loss_weight * local_loss

        if training:
            experiment.scaler.scale(loss).backward()
            experiment.scaler.unscale_(experiment.optimizer)
            torch.nn.utils.clip_grad_norm_(
                experiment.model.parameters(), experiment.config.grad_clip_norm
            )
            experiment.scaler.step(experiment.optimizer)
            experiment.scaler.update()

        losses.append(float(loss.detach().cpu()))
        targets.append(target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(output.logits).detach().cpu().numpy())
        del volumes, present, metadata, position, target, output
        del loss, combined_loss, local_loss

    return {
        "loss": float(np.mean(losses)),
        "target": np.concatenate(targets, axis=0),
        "probability": np.concatenate(probabilities, axis=0),
    }


def train_report_model(
    experiment: Experiment, supervision: ReportSupervision
) -> list[dict]:
    """Train on report labels, scoring on the held-out expert-gold studies."""
    for epoch in range(1, experiment.config.epochs + 1):
        started = time.time()
        train_result = run_report_epoch(
            experiment, experiment.train_loader, supervision, training=True
        )
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "seconds": round(time.time() - started, 1),
            "gate": float(np.abs(read_fusion_gate(experiment.model)).mean()),
        }

        if experiment.validation_loader is not None:
            held_out = run_report_epoch(
                experiment, experiment.validation_loader, supervision, training=False
            )
            scores = evaluate_predictions(held_out["target"], held_out["probability"])
            row["validation_loss"] = held_out["loss"]
            row["gold_macro_auc"] = scores["mean_auc"]
            row["per_target_auc"] = scores["per_target_auc"]

        experiment.history.append(row)
        auc = row.get("gold_macro_auc")
        print(
            f"epoch {epoch:>2} | train loss {row['train_loss']:.5f} | "
            f"gold AUC {('%.5f' % auc) if auc is not None else '   n/a  '} | "
            f"|gate| {row['gate']:.5f} | {row['seconds']}s"
        )
    return experiment.history
''',
)


# --- put B51 in the inherited flow -----------------------------------------
#
# B51's definitions go in front of the inherited training section, and the two
# cells that build and train are rewritten to use them. Leaving the inherited
# gold-only path in place would leave a trap: it still runs, it trains on the 58
# expert studies, and nothing about it announces that it is not B51.

TRAINING_SECTION = "Train on the extracted training subset and predict"

insert_cells(TRAINING_SECTION, DEFINITIONS)
retitle_cell(TRAINING_SECTION, "## 11. Train on the", "## 16. Train on the")
retitle_cell("Mandatory no-update memory", "### 11a.", "### 16a.")
retitle_cell("Train, plot losses, review cases", "### 11b.", "### 16b.")

rewrite_cell(
    "EXPERIMENT = build_experiment(PATHS, CONFIG)",
    "code",
    '''
# Build the report-supervised run. Report labels train the model; the expert-gold
# studies are held back so the score at the end means something.
EXPERIMENT, SUPERVISION = build_report_supervised_experiment(
    PATHS, CONFIG, adapt_hierarchy=True
)
''',
)

rewrite_cell(
    "RUN_TRAINING = False",
    "code",
    '''
# Keep training off until the preflight cell prints PASS.
RUN_TRAINING = False

if RUN_TRAINING:
    # Train on the report labels, scoring each epoch on the held-out gold studies.
    HISTORY = train_report_model(EXPERIMENT, SUPERVISION)
    # Plot the training and held-out loss curves.
    plot_loss_history(EXPERIMENT)
    # Display the numeric epoch history table.
    RESULTS = show_results(EXPERIMENT)

    print()
    print("final per-target AUC on the expert-gold studies")
    for target in TARGETS:
        value = HISTORY[-1].get("per_target_auc", {}).get(target)
        print(f"  {target:<28} {('%.5f' % value) if value is not None else 'n/a'}")

    # Plot up to twelve held-out gold MRI examples and their classifications.
    VALIDATION_CASE_TABLE = show_case_examples(
        EXPERIMENT, max_cases=12, title_prefix="Held-out gold"
    )
    # Build a local-DICOM loader for the separately extracted test subset.
    TEST_LOADER = build_test_loader(TEST_PATHS, CONFIG)
    # Generate one probability row and thresholded classification per test study.
    TEST_PREDICTIONS = predict_test_set(EXPERIMENT, TEST_LOADER)
    # Plot up to twelve unlabelled test MRI examples with their classifications.
    TEST_CASE_TABLE = show_case_examples(
        EXPERIMENT, loader=TEST_LOADER, max_cases=12, title_prefix="Test"
    )
    # Save this newly trained model, history, configuration, and test predictions.
    RUN_DIRECTORY = save_results(EXPERIMENT, test_predictions=TEST_PREDICTIONS)
''',
)


# --- the optional comparison ----------------------------------------------

append_cell(
    "markdown",
    """
## 17. Optional: watch the frozen and adapted settings side by side

B50 already answered this on the real data, so this section is off by default and
nothing below is needed to train the model. It is here for anyone who wants to
see the comparison happen rather than take it on trust.

One model is pretrained, its weights are snapshotted, and two runs continue from
that identical snapshot differing only in whether the hierarchy keeps learning.
Without the shared snapshot they would differ by their whole training history and
nothing could be attributed to the mechanism.

The **discordant pair fraction** is printed first. It is the share of study pairs
the two settings order differently, and an ROC AUC moves only on such pairs, so it
bounds how far the two scores could possibly differ. B48 and B49 were both judged
against a `+0.010` threshold their measurements could never reach — their ceilings
were `0.0015` and `0.0024` — and that was only noticed afterwards. Reading the
ceiling first is the habit that would have caught it.
""",
)

append_cell(
    "code",
    '''
def snapshot_weights(model: nn.Module) -> dict:
    """A detached copy of every weight, so both settings start identically."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def prepare_arm(experiment: Experiment, starting_weights: dict, adapt_hierarchy: bool) -> dict:
    """Reset one Experiment to the shared snapshot and set its freeze policy.

    The optimiser is rebuilt afterwards, never before: an optimiser holding a
    frozen tensor would still move it through weight decay.
    """
    experiment.model.load_state_dict(starting_weights)
    hierarchy_size = set_hierarchy_trainable(experiment.model, adapt_hierarchy)
    trainable = describe_trainable(experiment.model)

    if adapt_hierarchy and trainable["hierarchy"] == 0:
        raise RuntimeError("this arm should adapt the hierarchy but nothing is trainable")
    if not adapt_hierarchy and trainable["hierarchy"] != 0:
        raise RuntimeError("this arm should freeze the hierarchy but it is still trainable")

    groups = build_parameter_groups(experiment.model, experiment.config.learning_rate)
    experiment.optimizer = torch.optim.AdamW(
        groups, weight_decay=experiment.config.weight_decay
    )
    experiment.history = []
    return {
        "hierarchy_parameters": hierarchy_size,
        "trainable": trainable,
        "optimiser_groups": [group["name"] for group in groups],
    }


def run_one_arm(
    experiment: Experiment,
    supervision: ReportSupervision,
    starting_weights: dict,
    *,
    name: str,
    adapt_hierarchy: bool,
) -> dict:
    """Train one arm from the shared snapshot and score it on the gold studies."""
    print("=" * 62)
    print(f"{name}  (adapt_hierarchy={adapt_hierarchy})")
    print("=" * 62)

    setup = prepare_arm(experiment, starting_weights, adapt_hierarchy)
    print(f"hierarchy parameters : {setup['hierarchy_parameters']:,}")
    print(f"trainable            : {setup['trainable']}")
    print(f"optimiser groups     : {setup['optimiser_groups']}")

    train_report_model(experiment, supervision)

    if experiment.validation_loader is None:
        raise RuntimeError("this comparison needs expert-gold studies to score the arms")
    scored = run_report_epoch(
        experiment, experiment.validation_loader, supervision, training=False
    )
    gate = read_fusion_gate(experiment.model)
    print(f"|tanh(gate)| mean    : {np.abs(gate).mean():.5f}")

    return {
        "name": name,
        "gate": gate,
        "target": scored["target"],
        "probability": scored["probability"],
        "loss": scored["loss"],
        **setup,
    }
''',
)

append_cell(
    "code",
    '''
def discordant_pair_fraction(control: np.ndarray, candidate: np.ndarray) -> float:
    """Share of study pairs the two settings order differently, averaged over targets."""
    fractions = []
    for column in range(control.shape[1]):
        left, right = control[:, column], candidate[:, column]
        if len(left) < 2:
            fractions.append(0.0)
            continue
        upper = np.triu_indices(len(left), k=1)
        ls = np.sign(left[:, None] - left[None, :])[upper]
        rs = np.sign(right[:, None] - right[None, :])[upper]
        fractions.append(float(((ls * rs) < 0).sum() / len(ls)))
    return float(np.mean(fractions))


def compare_arms(frozen: dict, adapted: dict) -> dict:
    """Report the two settings side by side, with the ceiling read first."""
    if not np.array_equal(
        np.nan_to_num(frozen["target"], nan=-1.0),
        np.nan_to_num(adapted["target"], nan=-1.0),
    ):
        raise RuntimeError("the arms were scored on different studies; the pairing is broken")

    ceiling = discordant_pair_fraction(frozen["probability"], adapted["probability"])
    control = evaluate_predictions(frozen["target"], frozen["probability"])
    candidate = evaluate_predictions(adapted["target"], adapted["probability"])

    print()
    print(f"discordant pairs      {ceiling:.6f}")
    print("  the largest AUC difference this comparison could possibly show")
    print()
    print(f"{'setting':<30} {'macro AUC':>10} {'|gate|':>9} {'gold loss':>10}")
    for arm, scores in ((frozen, control), (adapted, candidate)):
        auc = scores["mean_auc"]
        shown = f"{auc:.6f}" if auc is not None else "undefined"
        print(
            f"{arm['name']:<30} {shown:>10} "
            f"{np.abs(arm['gate']).mean():>9.5f} {arm['loss']:>10.6f}"
        )

    delta = None
    if control["mean_auc"] is not None and candidate["mean_auc"] is not None:
        delta = candidate["mean_auc"] - control["mean_auc"]
        print()
        print(f"delta (adapted - frozen)  {delta:+.6f}")
        if abs(delta) > ceiling + 1e-9:
            print("WARNING: the delta exceeds its own ceiling; check the pairing")

    improved = [
        name
        for name in TARGETS
        if control["per_target_auc"].get(name) is not None
        and candidate["per_target_auc"].get(name) is not None
        and candidate["per_target_auc"][name] > control["per_target_auc"][name]
    ]
    print(f"targets improved          {len(improved)}/{len(TARGETS)}")

    print()
    print("On the real data B50 measured +0.011221 with a ceiling of 0.030652,")
    print("and all twelve targets improved. A subset this small is far noisier,")
    print("so read the direction and the gate rather than the third decimal.")
    return {
        "discordant_pair_fraction": ceiling,
        "control": control,
        "candidate": candidate,
        "delta": delta,
        "targets_improved": improved,
    }
''',
)

append_cell(
    "code",
    '''
RUN_B51_COMPARISON = False

if RUN_B51_COMPARISON:
    COMPARISON_EXPERIMENT, COMPARISON_SUPERVISION = build_report_supervised_experiment(
        PATHS, CONFIG, adapt_hierarchy=True
    )

    print("=" * 62)
    print("shared pretrain -- stands in for the Phase-9 base checkpoint")
    print("=" * 62)
    train_report_model(COMPARISON_EXPERIMENT, COMPARISON_SUPERVISION)
    STARTING_WEIGHTS = snapshot_weights(COMPARISON_EXPERIMENT.model)

    ARMS = {}
    for arm_name, adapt in (
        ("frozen_hierarchy_control", False),
        ("adapted_hierarchy_candidate", True),
    ):
        print()
        ARMS[arm_name] = run_one_arm(
            COMPARISON_EXPERIMENT,
            COMPARISON_SUPERVISION,
            STARTING_WEIGHTS,
            name=arm_name,
            adapt_hierarchy=adapt,
        )

    B51_COMPARISON = compare_arms(
        ARMS["frozen_hierarchy_control"], ARMS["adapted_hierarchy_candidate"]
    )
else:
    print("RUN_B51_COMPARISON is False. Section 16 is the one that trains the model.")
''',
)


def build(path: Path) -> Path:
    """Write the notebook."""
    cells = []
    for kind, text in CELLS:
        source = [line + "\n" for line in text.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
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
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = build(Path(__file__).with_name("b51_adapted_hierarchy_colab.ipynb"))
    print(f"{written}  ({len(CELLS)} cells)")
