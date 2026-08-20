"""Execute the notebook's definitions and check the model actually works.

A notebook that has never been run is a document, not code. These tests extract
every definition cell, execute them in one namespace, and then exercise the
parts that can run without the competition's DICOM files: the tensor shapes, the
loss, the metric, the submission writer, and -- most importantly -- that the
model produces the same answer as the deployed one on the cases that are easy to
get subtly wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

NOTEBOOK = Path(__file__).with_name("knee_mri_model.ipynb")

# The cells that read the competition's files. Everything before them is
# definitions and can be executed anywhere.
RUN_MARKERS = ("read_study_table(CONFIG.data_root", "pd.read_csv(LABEL_FILE)")


@pytest.fixture(scope="module")
def namespace():
    """Run every definition cell and hand back the resulting namespace."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so the executing namespace has to claim a module that actually exists.
    # In a real notebook that is `__main__`, which is what Colab gives it too.
    space: dict = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(marker in source for marker in RUN_MARKERS):
            break
        exec(compile(source, "<cell>", "exec"), space)
    return space


def test_the_notebook_is_valid_json_with_a_gpu_hint():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) > 30


def test_no_experiment_names_survive():
    """The whole point of this notebook: names describe function, not history."""
    import re

    text = NOTEBOOK.read_text(encoding="utf-8")
    # `b6`, `B20`, `b12_1`, `phase9` -- the archive's identifiers.
    pattern = re.compile(r"\b(?:[bB]\d{1,3}_\d+|[bB]\d{2,3}|[pP]hase\d+)\b")
    found = sorted(set(pattern.findall(text)))
    assert not found, f"experiment names leaked into the notebook: {found}"


def test_every_finding_is_present_and_ordered(namespace):
    findings = namespace["FINDINGS"]
    assert len(findings) == 12
    assert findings[0] == "ACL" and findings[-1] == "Fracture"
    assert "Baker's" in findings   # the apostrophe is part of the column name


def test_slice_positions_spread_through_a_volume(namespace):
    choose = namespace["choose_slice_positions"]
    positions = choose(30, 16, 1)
    assert len(positions) == 16
    assert positions.min() >= 0 and positions.max() <= 29
    assert (np.diff(positions) >= 0).all(), "positions must not go backwards"


def test_slice_positions_survive_a_series_shorter_than_the_sample(namespace):
    """A four-slice series still has to produce sixteen usable indices."""
    positions = namespace["choose_slice_positions"](4, 16, 1)
    assert len(positions) == 16
    assert positions.min() >= 0 and positions.max() <= 3


def test_the_offset_shifts_the_whole_comb(namespace):
    choose = namespace["choose_slice_positions"]
    base = choose(60, 16, 1, offset=0)
    shifted = choose(60, 16, 1, offset=2)
    assert (shifted >= base).all()
    assert (shifted[1:-1] - base[1:-1] == 2).all()


def test_intensity_scaling_lands_in_the_unit_range(namespace):
    scale = namespace["scale_intensities"]
    volume = np.random.default_rng(0).normal(500, 200, size=(8, 32, 32)).astype(np.float32)
    scaled = scale(volume)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    assert scaled.dtype == np.float32


def test_intensity_scaling_survives_a_bright_artefact(namespace):
    """One metal artefact must not compress the anatomy into a narrow band."""
    scale = namespace["scale_intensities"]
    volume = np.full((4, 16, 16), 100.0, dtype=np.float32)
    volume[0, 0, 0] = 1e6
    scaled = scale(volume)
    assert np.isfinite(scaled).all()
    assert scaled.max() <= 1.0


def test_the_crop_keeps_the_shape_and_changes_the_content(namespace):
    crop = namespace["crop_centre_and_resize"]
    volume = torch.randn(2, 3, 224, 224)
    cropped = crop(volume, 0.90)
    assert cropped.shape == volume.shape
    assert not torch.allclose(cropped, volume), "a 90% crop must change the image"


def test_the_batch_pads_to_the_widest_study(namespace):
    pad = namespace["pad_studies_into_batch"]
    items = [
        {
            "study_uid": f"study-{i}",
            "volumes": torch.zeros(k, 4, 3, 8, 8),
            "present": torch.ones(k),
            "series_meta": torch.ones(k, 3, dtype=torch.long),
        }
        for i, k in enumerate((3, 7))
    ]
    batch = pad(items)
    assert batch["volumes"].shape[:2] == (2, 7)
    assert batch["present"][0, 3:].eq(0).all(), "the shorter study must be padded off"
    assert batch["series_meta"][0, 3:].eq(0).all(), "padding must use the zero index"


def test_the_loss_ignores_unsupervised_cells(namespace):
    loss_fn = namespace["weighted_weak_bce"]
    logits = torch.zeros(2, 12, requires_grad=True)
    targets = torch.full((2, 12), 0.5)
    weights = torch.zeros(2, 12)
    weights[0, 0] = 1.0
    targets[0, 0] = 0.05

    multipliers = np.ones(12, dtype=np.float32)
    loss = loss_fn(logits, targets, weights, multipliers)
    loss.backward()
    grad = logits.grad
    assert grad[0, 0].abs() > 0, "the one supervised cell must produce gradient"
    assert grad[0, 1:].abs().sum() == 0, "unsupervised cells must produce none"


def test_a_batch_with_no_supervision_gives_zero_gradient_not_nan(namespace):
    """Studies whose report yielded nothing are kept, so this batch really occurs."""
    loss_fn = namespace["weighted_weak_bce"]
    logits = torch.zeros(2, 12, requires_grad=True)
    loss = loss_fn(logits, torch.full((2, 12), 0.5), torch.zeros(2, 12),
                   np.ones(12, dtype=np.float32))
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() == 0


def test_balance_multipliers_equalise_the_findings(namespace):
    multipliers = namespace["finding_balance_multipliers"]
    weights = np.zeros((100, 12), dtype=np.float32)
    weights[:, 0] = 1.0        # a well-covered finding
    weights[:10, 1:] = 1.0     # sparsely covered ones
    scaled = multipliers(weights)
    assert scaled[0] < scaled[1], "the sparse finding must be scaled up"
    mass = weights.sum(axis=0) * scaled
    assert np.allclose(mass, mass[0]), "every finding must end with equal mass"


def test_a_finding_with_no_supervision_is_refused(namespace):
    multipliers = namespace["finding_balance_multipliers"]
    weights = np.ones((4, 12), dtype=np.float32)
    weights[:, 5] = 0.0
    with pytest.raises(ValueError, match="no supervision"):
        multipliers(weights)


def test_the_metric_recognises_a_perfect_and_a_random_ranking(namespace):
    auc = namespace["binary_auc"]
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_a_finding_with_one_class_is_dropped_from_the_macro(namespace):
    """On a small evaluation set this is common, and averaging it as 0.5 would lie."""
    macro_fn = namespace["macro_roc_auc"]
    truth = np.zeros((10, 12))
    truth[:5, 0] = 1          # only the first finding has both classes
    score = np.random.default_rng(0).random((10, 12))
    macro, per_finding = macro_fn(truth, score)
    assert np.isfinite(macro)
    assert np.isnan(per_finding[1:]).all()
    assert macro == pytest.approx(per_finding[0])


def _tiny_model(namespace, **overrides):
    config = namespace["Config"](
        slices_per_series=2, image_size=32, slices_per_encoder_batch=4,
        study_layers=1, query_layers=1, dropout=0.0, **overrides,
    )
    return namespace["KneeAbnormalityModel"](config, pretrained_encoder=False), config


def test_the_model_produces_one_logit_per_finding(namespace):
    model, config = _tiny_model(namespace)
    model.eval()
    volumes = torch.randn(2, 3, config.slices_per_series, 3, 32, 32)
    present = torch.ones(2, 3)
    metadata = torch.ones(2, 3, 3, dtype=torch.long)
    with torch.no_grad():
        logits = model(volumes, present, metadata)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_a_study_with_no_readable_series_returns_the_learned_bias(namespace):
    """Attention over an entirely masked sequence is undefined; this is the guard."""
    model, config = _tiny_model(namespace)
    model.eval()
    volumes = torch.zeros(1, 2, config.slices_per_series, 3, 32, 32)
    with torch.no_grad():
        logits = model(volumes, torch.zeros(1, 2), torch.zeros(1, 2, 3, dtype=torch.long))
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[0], model.finding_bias)


def test_padded_series_do_not_change_the_answer(namespace):
    """Padding is an implementation detail and must be invisible to the output."""
    model, config = _tiny_model(namespace)
    model.eval()
    torch.manual_seed(0)
    real = torch.randn(1, 2, config.slices_per_series, 3, 32, 32)
    padded = torch.cat([real, torch.randn(1, 1, config.slices_per_series, 3, 32, 32)], dim=1)

    meta_real = torch.ones(1, 2, 3, dtype=torch.long)
    meta_padded = torch.cat([meta_real, torch.zeros(1, 1, 3, dtype=torch.long)], dim=1)

    with torch.no_grad():
        a = model(real, torch.ones(1, 2), meta_real)
        b = model(padded, torch.tensor([[1.0, 1.0, 0.0]]), meta_padded)
    assert torch.allclose(a, b, atol=1e-5)


def test_the_neighbour_context_is_bypassed_exactly_at_eval(namespace):
    """The scaffold shapes training and must not touch the deployed scoring."""
    model, config = _tiny_model(namespace)
    model.eval()
    torch.manual_seed(0)
    volumes = torch.randn(2, 2, config.slices_per_series, 3, 32, 32)
    present = torch.ones(2, 2)
    metadata = torch.ones(2, 2, 3, dtype=torch.long)

    with torch.no_grad():
        before = model(volumes, present, metadata)
        # Give the context real weights. At eval they must be unreachable.
        torch.nn.init.normal_(model.local_context.weight, std=1.0)
        after = model(volumes, present, metadata)
    assert torch.equal(before, after), "eval must not read the context weights"


def test_the_neighbour_context_does_reach_training(namespace):
    """Guard the guard: if it changed nothing in either mode it would be dead code."""
    model, config = _tiny_model(namespace)
    model.train()
    torch.manual_seed(0)
    volumes = torch.randn(2, 2, config.slices_per_series, 3, 32, 32)
    present = torch.ones(2, 2)
    metadata = torch.ones(2, 2, 3, dtype=torch.long)

    with torch.no_grad():
        before = model(volumes, present, metadata)
        torch.nn.init.normal_(model.local_context.weight, std=1.0)
        after = model(volumes, present, metadata)
    assert not torch.equal(before, after), "training must use the context"


def test_the_second_pooling_route_starts_switched_off(namespace):
    """The gate is zero-initialised, so route B enters only if training wants it."""
    model, _ = _tiny_model(namespace)
    assert torch.equal(model.summary_gate, torch.zeros_like(model.summary_gate))
    assert torch.tanh(model.summary_gate).abs().sum() == 0


def test_unfreezing_frees_the_output_end_only(namespace):
    model, _ = _tiny_model(namespace)
    namespace["freeze_encoder"](model)
    assert not any(p.requires_grad for p in model.encoder.parameters())

    freed = namespace["unfreeze_last_encoder_blocks"](model, 1)
    assert freed > 0
    trainable = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
    assert all(n.startswith(("pre_classifier", "features.7")) for n in trainable)
    assert not any(n.startswith("features.0") for n in trainable)


def test_the_encoder_stays_in_eval_after_unfreezing(namespace):
    """Deliberate: only gradient flow changes, the forward pass does not."""
    model, _ = _tiny_model(namespace)
    namespace["freeze_encoder"](model)
    namespace["unfreeze_last_encoder_blocks"](model, 1)
    assert not model.encoder.training


def test_the_encoder_group_learns_far_more_slowly(namespace):
    model, config = _tiny_model(namespace)
    namespace["freeze_encoder"](model)
    namespace["unfreeze_last_encoder_blocks"](model, 1)
    groups = namespace["build_parameter_groups"](model, config)
    rates = {g["name"]: g["lr"] for g in groups}
    assert rates["head"] == config.head_lr
    assert rates["encoder"] == pytest.approx(config.head_lr * config.encoder_lr_fraction)


def test_a_frozen_encoder_produces_only_one_group(namespace):
    model, config = _tiny_model(namespace)
    namespace["freeze_encoder"](model)
    groups = namespace["build_parameter_groups"](model, config)
    assert [g["name"] for g in groups] == ["head"]


def test_the_submission_has_the_exact_expected_columns(namespace, tmp_path):
    write = namespace["write_submission"]
    uids = ["a", "b", "c"]
    frame = write(uids, np.full((3, 12), 0.5), path=tmp_path / "submission.csv")
    assert list(frame.columns) == ["StudyInstanceUID", *namespace["FINDINGS"]]
    assert frame["StudyInstanceUID"].tolist() == uids
    assert (tmp_path / "submission.csv").is_file()


def test_a_probability_outside_the_unit_range_is_refused(namespace, tmp_path):
    write = namespace["write_submission"]
    bad = np.full((2, 12), 0.5)
    bad[0, 0] = 1.5
    with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
        write(["a", "b"], bad, path=tmp_path / "submission.csv")


def test_a_repeated_study_is_refused(namespace, tmp_path):
    write = namespace["write_submission"]
    with pytest.raises(AssertionError, match="twice"):
        write(["a", "a"], np.full((2, 12), 0.5), path=tmp_path / "submission.csv")
