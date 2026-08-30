"""Generate the B51-shaped Google Colab subset notebook.

B51 asks one question: should the study hierarchy -- the part of the model that
turns encoded slices into twelve answers -- be allowed to learn, rather than
staying frozen while only the encoder tail and a small local head train?

On the real data B50 answered it. A matched pair differing in nothing else gave
`+0.011221` macro AUC on 548 unseen-scanner studies, all twelve targets
improved, and the effect was larger on the base path (`+0.011676`) than on the
combined one -- exactly where the mechanism says it should be.

This notebook reproduces that **structure** on a Google Drive subset. It is not
the B51 protocol in ``developments/``: there is no 4,349-study population, no
scanner-grouped gate, no report-derived weak supervision, and the weights start
from scratch rather than from the Phase-9 checkpoint. What it does faithfully is
the shape of the comparison:

    pretrain one model briefly            the shared starting point, standing in
                                          for the Phase-9 base checkpoint
    arm A: hierarchy frozen               what every model from B37 to B49 did
    arm B: hierarchy adapts at 0.05x      what B50 validated

Both arms start from the same snapshot and differ in nothing else, so a
difference between them is attributable. The notebook also prints the fusion
gate for each arm, because on the real data that number moved first: the adapted
arm settled at roughly half the frozen arm's, which is what an improved base
looks like from the head's point of view.

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
    CELLS.append((kind, text.strip("\n")))


# --- what this notebook is -------------------------------------------------

replace_cell(
    1,
    "markdown",
    """
# B51 — does the study hierarchy help when it is allowed to learn?

Every model in this project from B37 to B49 froze the study hierarchy: the
slice-pooling, the study aggregation and the twelve pathology outputs stayed
exactly as an earlier run left them, while only the encoder tail and a small
local head trained.

Those nine experiments moved the score very little. The measured reason is that
they were all refining a branch that reaches the prediction through a gate worth
about 2%, so roughly 98% of every score came from weights that had not received
a gradient in a long time.

B50 tested the obvious alternative on the real data and it was supported:

| surface | control | candidate | delta |
|---|---|---|---|
| combined | 0.763117 | 0.774336 | **+0.011221** |
| base | 0.762566 | 0.774243 | +0.011676 |
| local | 0.743541 | 0.753820 | +0.010278 |

All twelve targets improved, and the effect was **larger on the base path than
on the combined one** — precisely where a change to the hierarchy should land.

This notebook reproduces that comparison in miniature on a Drive subset.
""",
)

replace_cell(
    2,
    "markdown",
    """
## What this notebook is, and is not

**It is** a runnable, self-contained version of B51's comparison. One model is
pretrained briefly to make a shared starting point, then two arms continue from
that snapshot and differ in exactly one thing:

```text
arm A   hierarchy frozen        what B37 through B49 all did
arm B   hierarchy adapts        at 0.05x the head's learning rate
```

**It is not** the B51 protocol in `developments/`. That one trains on all 4,349
report-only studies, starts from the Phase-9 checkpoint, uses report-derived
weak labels and a frozen scanner-grouped validation gate, and takes about eight
and a half hours on an RTX A4500. This notebook starts from random weights on
whatever subset fits your Drive, so its absolute numbers mean nothing.

What transfers is the **shape** of the result: whether letting the hierarchy
learn helps, measured against an otherwise identical control.

Watch two things:

1. the macro AUC difference between the arms;
2. the **fusion gate**. On the real data the adapted arm's gate settled at
   roughly half the frozen arm's, before any score was computed. When the base
   improves, the model leans on the local correction less.
""",
)


# --- the B51 mechanism -----------------------------------------------------

append_cell(
    "markdown",
    """
## 12. B51: freeze or adapt the study hierarchy

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
model from B37 to B49 did. Adapting it means they learn, slowly.

Two details below are the ones that are easy to get wrong, and both are the same
mistakes the real implementation had to avoid.

**Gradients are not training mode.** `requires_grad` and `train()`/`eval()` are
independent. Freezing must be done with `requires_grad_(False)`, never by
switching modules to eval, which would also change dropout and normalisation and
make the arms differ in more than one way.

**A frozen parameter must not reach the optimiser.** If it did, weight decay
would still move it, and the "frozen" arm would not be frozen.
""",
)

append_cell(
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
    arms would then differ in more than the one thing being tested.
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
    decay, so the frozen arm would not actually be frozen. Only tensors that
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
    """What is actually learning, so an arm can be checked rather than assumed."""
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


# --- the comparison --------------------------------------------------------

append_cell(
    "markdown",
    """
## 13. Run both arms from one shared starting point

The order matters. One model is trained first and its weights are snapshotted.
Both arms then start from that identical snapshot, so the only difference
between them is whether the hierarchy keeps learning.

Without the shared snapshot the arms would differ by their whole training
history and nothing could be attributed to the mechanism.
""",
)

append_cell(
    "code",
    '''
def snapshot_weights(model: nn.Module) -> dict:
    """A detached copy of every weight, so both arms start identically."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def prepare_arm(experiment: Experiment, starting_weights: dict, adapt_hierarchy: bool) -> dict:
    """Reset one Experiment to the shared snapshot and set its freeze policy.

    The optimiser is rebuilt afterwards, never before: an optimiser holding a
    frozen tensor would still move it through weight decay, and the frozen arm
    would not be frozen.
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
    experiment: Experiment, starting_weights: dict, *, name: str, adapt_hierarchy: bool
) -> dict:
    """Train one arm from the shared snapshot and score it on the held-out split."""
    print("=" * 62)
    print(f"{name}  (adapt_hierarchy={adapt_hierarchy})")
    print("=" * 62)

    setup = prepare_arm(experiment, starting_weights, adapt_hierarchy)
    print(f"hierarchy parameters : {setup['hierarchy_parameters']:,}")
    print(f"trainable            : {setup['trainable']}")
    print(f"optimiser groups     : {setup['optimiser_groups']}")

    history = train_model(experiment)

    if experiment.validation_loader is None:
        raise RuntimeError("this comparison needs a validation split to score the arms")
    scored = run_epoch(experiment, experiment.validation_loader, training=False)
    gate = read_fusion_gate(experiment.model)
    print(f"|tanh(gate)| mean    : {np.abs(gate).mean():.5f}")

    return {
        "name": name,
        "history": history,
        "gate": gate,
        "target": scored["target"],
        "probability": scored["probability"],
        "loss": scored["loss"],
        **setup,
    }
''',
)


append_cell(
    "markdown",
    """
## 14. Compare the arms, ceiling first

The **discordant pair fraction** is the share of study pairs the two arms order
differently. An ROC AUC moves only on such pairs, so that number bounds how far
the arms' scores could possibly differ.

It is printed before the difference, not after. On the real data B48 and B49
were both judged against a `+0.010` threshold their measurements could not
reach — their ceilings were `0.0015` and `0.0024` — and that was only noticed
afterwards. Checking it first is the habit that would have caught it.
""",
)

append_cell(
    "code",
    '''
def discordant_pair_fraction(control: np.ndarray, candidate: np.ndarray) -> float:
    """Share of study pairs the two arms order differently, averaged over targets."""
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
    """Report the two arms side by side, with the ceiling read first."""
    if not np.array_equal(frozen["target"], adapted["target"]):
        raise RuntimeError("the arms were scored on different studies; the pairing is broken")

    ceiling = discordant_pair_fraction(frozen["probability"], adapted["probability"])
    control = evaluate_predictions(frozen["target"], frozen["probability"])
    candidate = evaluate_predictions(adapted["target"], adapted["probability"])

    print()
    print(f"discordant pairs      {ceiling:.6f}")
    print("  the largest AUC difference this comparison could possibly show")
    print()
    print(f"{'arm':<30} {'macro AUC':>10} {'|gate|':>9} {'val loss':>10}")
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
        for name in TARGET_COLUMNS
        if control["per_target_auc"].get(name) is not None
        and candidate["per_target_auc"].get(name) is not None
        and candidate["per_target_auc"][name] > control["per_target_auc"][name]
    ]
    print(f"targets improved          {len(improved)}/{len(TARGET_COLUMNS)}")

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
    "markdown",
    """
## 15. Execute the comparison

Set `RUN_B51_COMPARISON = True` only after the preflight in section 11a has
printed `PASS`. Three training runs happen here — one shared pretrain and two
arms — so allow roughly three times a single run.
""",
)

append_cell(
    "code",
    '''
RUN_B51_COMPARISON = False

if RUN_B51_COMPARISON:
    print("=" * 62)
    print("shared pretrain -- stands in for the Phase-9 base checkpoint")
    print("=" * 62)
    set_hierarchy_trainable(EXPERIMENT.model, True)
    EXPERIMENT.optimizer = torch.optim.AdamW(
        build_parameter_groups(EXPERIMENT.model, EXPERIMENT.config.learning_rate),
        weight_decay=EXPERIMENT.config.weight_decay,
    )
    train_model(EXPERIMENT)
    STARTING_WEIGHTS = snapshot_weights(EXPERIMENT.model)

    ARMS = {}
    for arm_name, adapt in (
        ("frozen_hierarchy_control", False),
        ("adapted_hierarchy_candidate", True),
    ):
        print()
        ARMS[arm_name] = run_one_arm(
            EXPERIMENT, STARTING_WEIGHTS, name=arm_name, adapt_hierarchy=adapt
        )

    B51_COMPARISON = compare_arms(
        ARMS["frozen_hierarchy_control"], ARMS["adapted_hierarchy_candidate"]
    )
else:
    print("RUN_B51_COMPARISON is False. Set it to True after the preflight passes.")
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
    print(f"{written} ({len(CELLS)} cells)")
