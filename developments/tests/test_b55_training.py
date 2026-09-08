"""B55's trainer, and the two hooks it needed from B52.

B53 copied B52's loop to change one thing, and the copy is now several hundred
lines that have to be kept in step by hand. B55 does not: `train_b52` gained a
`dataset_factory` and an `identity`, both defaulting to B52's own, so B55 is
thin and a later correction to the loop reaches both runs.

The tests here are mostly about that defaulting. A hook that changed B52's
behaviour when nobody passed it would silently alter every completed run's
successor.
"""

from __future__ import annotations

import inspect

import pytest

from rsna_knee.b42_constant_area_aspect_sparse_mil import B42ConstantAreaAspectDataset
from rsna_knee.b52_competition_training import (
    B52_EXPERIMENT,
    B52_VERSION,
    _build_dataset,
    train_b52,
)
from rsna_knee.b55_physical_geometry import (
    B55_REFERENCE_SIDE,
    B55PhysicalGeometryDataset,
)
from rsna_knee.b55_physical_geometry_training import (
    B55_DEFAULT_EPOCHS,
    B55_EXPERIMENT,
    B55_VERSION,
    b55_dataset_factory,
    train_b55,
)
from rsna_knee.physical_crop import CROP_MM


# --- the hooks default to B52's own behaviour ---------------------------------


def test_both_hooks_default_to_none():
    """So every B52 run made before they existed is byte-identical."""
    parameters = inspect.signature(train_b52).parameters

    assert parameters["dataset_factory"].default is None
    assert parameters["identity"].default is None


def test_the_default_factory_is_b52s_builder():
    source = inspect.getsource(train_b52)
    assert "build_dataset = dataset_factory or _build_dataset" in source


def test_the_default_identity_is_b52s_own():
    source = inspect.getsource(train_b52)
    assert '"experiment": B52_EXPERIMENT' in source
    assert '"version": B52_VERSION' in source


def test_both_dataset_sites_go_through_the_hook():
    """The validation loader too: B55's geometry must reach what it scores."""
    source = inspect.getsource(train_b52)

    assert source.count("build_dataset(") == 2
    assert "_build_dataset(" not in source.split("build_dataset = ")[1], (
        "a construction site still bypasses the hook"
    )


def test_the_checkpoint_writes_the_resolved_identity():
    source = inspect.getsource(train_b52)
    assert '"experiment": named["experiment"]' in source
    assert '"version": named["version"]' in source


# --- the factory ---------------------------------------------------------------


def test_the_factory_matches_the_builder_it_replaces():
    """A mismatched signature would fail inside train_b52, hours in."""
    built = b55_dataset_factory(CROP_MM, B55_REFERENCE_SIDE**2)

    assert list(inspect.signature(built).parameters) == list(
        inspect.signature(_build_dataset).parameters
    )


def test_the_factory_builds_b55s_dataset():
    built = b55_dataset_factory(CROP_MM, B55_REFERENCE_SIDE**2)
    source = inspect.getsource(built)

    assert "B55PhysicalGeometryDataset" in source
    assert issubclass(B55PhysicalGeometryDataset, B42ConstantAreaAspectDataset), (
        "every frozen contract that tests for the B42 dataset must still hold"
    )


def test_the_factory_carries_the_crop_and_the_resolution():
    source = inspect.getsource(b55_dataset_factory(CROP_MM, B55_REFERENCE_SIDE**2))
    assert "crop_mm=float(crop_mm)" in source
    assert "reference_area=int(reference_area)" in source


def test_the_factory_refuses_the_spacing_conditioning():
    """It was tested at 12.65% of its own sum and measured -0.004908. Closed."""
    built = b55_dataset_factory(CROP_MM, B55_REFERENCE_SIDE**2)

    with pytest.raises(ValueError, match="does not carry the spacing"):
        built(None, None, None, None, None, None, spacing=True)


# --- B55's identity, which B54 did not have -----------------------------------


def test_b55_names_itself_rather_than_borrowing_b52s():
    """B54's checkpoint says B52 and that is still an open cleanup task."""
    assert B55_EXPERIMENT != B52_EXPERIMENT
    assert B55_VERSION != B52_VERSION
    assert "B55" in B55_EXPERIMENT


def test_the_trainer_passes_its_own_identity():
    source = inspect.getsource(train_b55)
    assert 'identity={"experiment": B55_EXPERIMENT, "version": B55_VERSION}' in source


# --- what B55 inherits rather than restates -----------------------------------


def test_the_trainer_calls_b52_rather_than_copying_it():
    """B53 copied the loop; B55 must not, or the two drift."""
    source = inspect.getsource(train_b55)

    assert "return train_b52(" in source
    assert "for epoch in range" not in source, "B55 must not own an epoch loop"
    assert "save_checkpoint" not in source


def test_the_rates_and_stages_are_b52s_constants():
    source = inspect.getsource(train_b55)
    for name in (
        "B52_DEFAULT_ENCODER_STAGES",
        "B52_DEFAULT_ENCODER_LR_SCALE",
        "B52_DEFAULT_HIERARCHY_LR_SCALE",
    ):
        assert name in source, f"{name} must be inherited, not restated"


def test_the_default_schedule_is_eight_epochs():
    assert B55_DEFAULT_EPOCHS == 8


def test_the_defaults_are_the_measured_ones():
    parameters = inspect.signature(train_b55).parameters

    assert parameters["crop_mm"].default == CROP_MM == 130.0
    assert parameters["reference_side"].default == B55_REFERENCE_SIDE == 336
    assert parameters["all_data"].default is True


def test_the_supervision_guard_is_passed_through_not_defeated():
    """B55 must be able to declare a new count, but not skip the check."""
    parameters = inspect.signature(train_b55).parameters
    assert parameters["expected_supervision_cells"].default is None

    source = inspect.getsource(train_b55)
    assert "expected_supervision_cells=expected_supervision_cells" in source


def test_the_teacher_is_not_wired_in():
    """It arrives through --labels-root, so the two decisions stay separable."""
    source = inspect.getsource(train_b55)
    assert "rebuild(" not in source
    assert "apply_rubric" not in source


# --- the command line ---------------------------------------------------------


def test_the_command_line_renders(capsys, monkeypatch):
    import sys

    from rsna_knee import b55_physical_geometry_training as b55

    monkeypatch.setattr(sys, "argv", ["b55", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        b55.main()

    assert exit_code.value.code == 0
    printed = capsys.readouterr().out
    for flag in ("--crop-mm", "--reference-side", "--num-workers", "--gate-split"):
        assert flag in printed


def test_all_data_is_the_default_and_gate_split_is_the_opt_out():
    source = inspect.getsource(
        __import__(
            "rsna_knee.b55_physical_geometry_training", fromlist=["main"]
        ).main
    )
    assert "all_data=not args.gate_split" in source


# --- augmentation, composed rather than reimplemented -------------------------
#
# B55 originally had none, which would have discarded B53's finding if B53 turns
# out positive. It is off by default: whether augmentation helps at these
# settings is what B53 is currently measuring, and bundling a fourth unvalidated
# change would be the opposite of what this endpoint is for.


def test_augmentation_is_off_by_default():
    assert inspect.signature(train_b55).parameters["augment"].default is False


def test_the_augmented_dataset_composes_both_behaviours():
    from rsna_knee.b53_augmented_training import B53AugmentedDataset
    from rsna_knee.b55_physical_geometry_training import B55AugmentedDataset

    assert issubclass(B55AugmentedDataset, B53AugmentedDataset)
    assert issubclass(B55AugmentedDataset, B55PhysicalGeometryDataset)


def test_the_resolution_order_augments_after_b55_builds_the_tensor():
    """B53 augments in __getitem__, B55 produces the tensor in _load_b42.

    If B55 came first in the MRO its __getitem__ would win and the
    augmentation would never run -- silently, since the pixels would still be
    valid B55 pixels.
    """
    from rsna_knee.b53_augmented_training import B53AugmentedDataset
    from rsna_knee.b55_physical_geometry_training import B55AugmentedDataset

    order = list(B55AugmentedDataset.__mro__)
    assert order.index(B53AugmentedDataset) < order.index(B55PhysicalGeometryDataset)
    assert B55AugmentedDataset.__getitem__ is B53AugmentedDataset.__getitem__


def test_b53s_loader_delegates_down_to_b55s_geometry(monkeypatch):
    """Both override `_load_b42`, so this could easily not be true.

    B53's shifts the slice centres and then calls `super()._load_b42`. Under
    this MRO that `super()` is B55's, so the physical crop still runs. Proved by
    running it rather than by reading it: if B53 ever stopped delegating, the
    pixels would still be valid B42 pixels and nothing would look wrong.
    """
    from rsna_knee.b55_physical_geometry import B55PhysicalGeometryDataset
    from rsna_knee.b55_physical_geometry_training import B55AugmentedDataset

    reached = []
    monkeypatch.setattr(
        B55PhysicalGeometryDataset,
        "_load_b42",
        lambda self, uid, series_uid, plane: reached.append((uid, plane)),
    )

    # A real instance, not a stand-in: zero-argument `super()` binds to the
    # object's own type, so a stub cannot exercise the delegation at all.
    instance = B55AugmentedDataset.__new__(B55AugmentedDataset)
    instance.slice_jitter = 0
    instance._draw = None

    instance._load_b42("study", "series", "sagittal")
    assert reached == [("study", "sagittal")], "B55's geometry was bypassed"


def test_validation_is_never_augmented():
    """Augmenting it would change what the score measures, not what is learnt."""
    from rsna_knee.b53_augmented_training import AugmentationPolicy
    from rsna_knee.b55_physical_geometry_training import B55AugmentedDataset

    policy = AugmentationPolicy(rotation_deg=5.0)
    source = inspect.getsource(b55_dataset_factory)

    assert "if train and policy is not None" in source
    assert "B55AugmentedDataset" in source
    assert B55AugmentedDataset is not B55PhysicalGeometryDataset


def test_the_factory_contract_carries_the_train_flag():
    from rsna_knee.b52_competition_training import _build_dataset

    assert "train" in inspect.signature(_build_dataset).parameters
    assert inspect.signature(_build_dataset).parameters["train"].default is False


def test_only_the_training_surface_is_told_it_is_training():
    source = inspect.getsource(train_b52)
    assert source.count("train=True,") == 1, "validation must not be built as train"


def test_the_loop_advances_the_augmentation_draw():
    """Without this every epoch repeats one draw: augmentation that reaches the
    pixels and then stops varying."""
    source = inspect.getsource(train_b52)
    assert 'hasattr(train_dataset, "set_epoch")' in source
    assert "train_dataset.set_epoch(epoch)" in source


def test_asking_for_augmentation_that_is_all_zero_is_refused():
    """The B52 failure: a flag that sets fields nobody reads."""
    from rsna_knee.b55_physical_geometry_training import train_b55

    with pytest.raises(ValueError, match="every configured value is zero"):
        train_b55(
            {"b7_rotation_deg": 0.0, "b7_translate_frac": 0.0, "b7_scale_jitter": 0.0,
             "b7_gamma_jitter": 0.0, "b7_bias_field_strength": 0.0,
             "b7_noise_std": 0.0, "b7_slice_dropout": 0.0},
            data_root=".", labels_root=".", series_policy_path=".",
            base_checkpoint=".", domain_split=".", augment=True,
        )


# --- what the checkpoint says the run did -------------------------------------
#
# `train_b52`'s own `augment` defaults to **True**. B55 originally omitted it,
# so a run with augmentation off would have written `augmentation_enabled: true`
# into its checkpoint over undistorted pixels -- exactly the B52 failure that
# B53 exists to correct, reappearing one layer up and just as invisible.
#
# These run `train_b55` with `train_b52` replaced, so they read the arguments
# that were actually passed rather than the text of the call.


def _captured_call(monkeypatch, **kwargs) -> dict:
    from rsna_knee import b55_physical_geometry_training as b55

    seen: dict = {}

    def fake_train_b52(config, **passed):
        seen.update(passed)
        return None

    monkeypatch.setattr(b55, "train_b52", fake_train_b52)
    b55.train_b55(
        {},
        data_root=".",
        labels_root=".",
        series_policy_path=".",
        base_checkpoint=".",
        domain_split=".",
        **kwargs,
    )
    return seen


@pytest.mark.parametrize("augment", [False, True])
def test_the_augment_flag_reaches_b52_rather_than_defaulting(monkeypatch, augment):
    """Omitting it wrote `augmentation_enabled: true` over untouched pixels."""
    assert inspect.signature(train_b52).parameters["augment"].default is True, (
        "the whole hazard is that B52's default is True"
    )

    assert _captured_call(monkeypatch, augment=augment)["augment"] is augment


def test_the_recorded_policy_matches_the_flag(monkeypatch):
    """A boolean can lie. The policy that was built is written instead."""
    off = _captured_call(monkeypatch, augment=False)
    assert off["extra"]["b55_augmentation"] is None

    on = _captured_call(monkeypatch, augment=True)
    assert on["extra"]["b55_augmentation"], "an active policy, not an empty dict"
    assert all(value > 0 for value in on["extra"]["b55_augmentation"].values())


def test_the_geometry_is_recorded_as_it_was_used(monkeypatch):
    recorded = _captured_call(monkeypatch, crop_mm=150.0, reference_side=224)["extra"]

    assert recorded["b55_geometry"]["crop_mm"] == 150.0
    assert recorded["b55_geometry"]["reference_side"] == 224
    assert recorded["b55_geometry"]["reference_area"] == 224 * 224


def test_b55s_extra_fields_are_named_for_b55(monkeypatch):
    """So they cannot collide with, or be mistaken for, B52's own record."""
    recorded = _captured_call(monkeypatch, augment=False)["extra"]

    assert set(recorded) == {"b55_geometry", "b55_augmentation"}
    assert all(name.startswith("b55_") for name in recorded)


# --- the hook cannot rewrite B52's record -------------------------------------


def test_extra_cannot_overwrite_what_b52_says_about_the_run():
    """A subclass silently rewriting `augmentation_enabled` is the same bug."""
    from rsna_knee.b52_competition_training import _refuse_reserved_extra

    with pytest.raises(ValueError, match="augmentation_enabled"):
        _refuse_reserved_extra({"augmentation_enabled": False})


def test_extra_may_add_fields_of_its_own():
    from rsna_knee.b52_competition_training import _merge_extra, _refuse_reserved_extra

    _refuse_reserved_extra({"b55_geometry": {"crop_mm": 130.0}})
    payload = {"experiment": "B55", "augmentation_enabled": True}

    assert _merge_extra(payload, {"b55_augmentation": None})["b55_augmentation"] is None


def test_the_merge_is_the_backstop_for_the_frozen_set():
    """It reads the payload that was built, so a new field is protected too."""
    from rsna_knee.b52_competition_training import _merge_extra

    with pytest.raises(ValueError, match="added_later"):
        _merge_extra({"added_later": 1}, {"added_later": 2})


def test_the_reserved_names_are_fields_the_checkpoint_really_writes():
    """Otherwise the frozen set rots into a list of names nothing protects."""
    import ast

    from rsna_knee.b52_competition_training import B52_RESERVED_PAYLOAD_FIELDS

    tree = ast.parse(inspect.getsource(train_b52).lstrip())
    written = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    missing = sorted(B52_RESERVED_PAYLOAD_FIELDS - written)
    assert not missing, f"reserved but never written: {missing}"


def test_augmentation_enabled_is_reserved():
    """The one field this whole guard exists for."""
    from rsna_knee.b52_competition_training import B52_RESERVED_PAYLOAD_FIELDS

    assert "augmentation_enabled" in B52_RESERVED_PAYLOAD_FIELDS
    assert "experiment" in B52_RESERVED_PAYLOAD_FIELDS
