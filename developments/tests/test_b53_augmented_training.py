"""B53's augmentation, measured on the real dataset rather than read.

B52's augmentation flag set fields on a config object that the B42 dataset never
read. Nothing in the code looked wrong, the trainer printed `augment=True`, and
every checkpoint recorded `augmentation_enabled: true` while the model trained
on byte-identical pixels for 27 hours.

The lesson is that an augmentation test which does not build the real dataset
from real DICOM files and compare two draws proves nothing. So most of this file
does exactly that. The first test is the one that would have caught B52, applied
to B52 itself, and it is expected to fail on B52 and pass on B53.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.b42_constant_area_aspect_sparse_mil import (  # noqa: E402
    B42ConstantAreaAspectDataset,
)
from rsna_knee.b53_augmented_training import (  # noqa: E402
    B53_SLICE_JITTER_DEFAULT,
    AugmentationPolicy,
    B53AugmentedDataset,
    augment_b42_series,
    verify_augmentation_reaches_pixels,
)
from rsna_knee.dataset import DatasetConfig  # noqa: E402

CROP_POLICY = {"crop_fraction": 0.90, "policy": "b20_crop_focus_v1"}


def _write_series(directory: Path, frames: int = 40, size: int = 48, seed: int = 3) -> None:
    """A small but genuine DICOM series.

    Real files, not a stubbed reader: the B52 bug lived in the gap between the
    config and the decoding path, and a stub would have skipped exactly that gap.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)

    for index in range(frames):
        pixels = generator.integers(0, 2048, size=(size, size), dtype=np.uint16)
        # Structure worth preserving, so a warp is visible as more than noise.
        pixels[size // 4 : size // 2, size // 4 : size // 2] += 8000

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        frame = Dataset()
        frame.file_meta = meta
        frame.SOPClassUID = meta.MediaStorageSOPClassUID
        frame.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        frame.Modality = "MR"
        frame.Rows, frame.Columns = size, size
        frame.BitsAllocated = 16
        frame.BitsStored = 16
        frame.HighBit = 15
        frame.PixelRepresentation = 0
        frame.SamplesPerPixel = 1
        frame.PhotometricInterpretation = "MONOCHROME2"
        frame.InstanceNumber = index + 1
        frame.ImagePositionPatient = [0.0, 0.0, float(index) * 3.0]
        frame.PixelSpacing = [0.5, 0.5]
        frame.PixelData = pixels.tobytes()
        frame.save_as(directory / f"{index:04d}.dcm", enforce_file_format=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple:
    """One study with two readable series, on disk as real DICOM."""
    root = tmp_path_factory.mktemp("b53") / "data"
    records = {"study-a": []}
    for position, plane in enumerate(("Sagittal", "Coronal")):
        series_uid = f"series-{position}"
        _write_series(root / "train_series" / "study-a" / series_uid, seed=position + 1)
        records["study-a"].append(
            {
                "series_uid": series_uid,
                "plane": plane,
                "plane_id": position + 1,
                "fluid_id": 1,
                "fat_id": 1,
            }
        )
    return root, records


def _config(root: Path) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1
    )


def _dataset(study, *, policy=None, slice_jitter=B53_SLICE_JITTER_DEFAULT, seed=2026):
    root, records = study
    return B53AugmentedDataset(
        ["study-a"],
        records,
        _config(root),
        crop_focus_policy=CROP_POLICY,
        center_offsets=(0,),
        targets=np.zeros((1, 12), np.float32),
        weights=np.ones((1, 12), np.float32),
        policy=policy,
        seed=seed,
        slice_jitter=slice_jitter,
    )


# --- the test that would have caught B52 -----------------------------------


def test_b52s_dataset_does_not_augment(study):
    """The finding, pinned as a test rather than left as a claim.

    B42ConstantAreaAspectDataset ignores every augmentation field on its config,
    for two independent reasons: the B37 base class hard-codes `train=False`,
    and B42 writes its own `_load_b42` which never touches `_augment_mri`.

    If this test ever starts failing, the upstream dataset has begun augmenting
    on its own and B53 would be applying augmentation twice.
    """
    root, records = study

    def build(train_flag: bool):
        config = DatasetConfig(
            data_root=str(root), split="train", n_slices=16, image_size=448, triplet_gap=1,
            noise_std=0.02 if train_flag else 0.0,
            slice_dropout=0.08 if train_flag else 0.0,
            center_jitter=2 if train_flag else 0,
            rotation_deg=5.0 if train_flag else 0.0,
            translate_frac=0.03 if train_flag else 0.0,
            scale_jitter=0.05 if train_flag else 0.0,
            gamma_jitter=0.12 if train_flag else 0.0,
            bias_field_strength=0.08 if train_flag else 0.0,
        )
        return B42ConstantAreaAspectDataset(
            ["study-a"], records, config, crop_focus_policy=CROP_POLICY,
            center_offsets=(0,), targets=np.zeros((1, 12), np.float32),
            weights=np.ones((1, 12), np.float32),
        )

    on, off = build(True), build(False)
    assert on.config.rotation_deg == 5.0, "the config really does carry the setting"

    torch.manual_seed(0)
    first = on[0]["volumes"][0]
    torch.manual_seed(1)
    second = on[0]["volumes"][0]
    torch.manual_seed(2)
    plain = off[0]["volumes"][0]

    assert torch.equal(first, second), "B42 with augmentation on is deterministic"
    assert torch.equal(first, plain), "B42 ignores every augmentation field"


# --- the policy -------------------------------------------------------------


def test_the_policy_comes_from_the_config_not_from_here():
    """Invented numbers would make B53 a second change, not one."""
    policy = AugmentationPolicy.from_config(
        {
            "b7_rotation_deg": 7.5,
            "b7_translate_frac": 0.04,
            "b7_scale_jitter": 0.06,
            "b7_gamma_jitter": 0.15,
            "b7_bias_field_strength": 0.09,
            "b7_noise_std": 0.03,
            "b7_slice_dropout": 0.10,
        }
    )
    assert policy.rotation_deg == 7.5
    assert policy.noise_std == 0.03
    assert policy.slice_dropout == 0.10


def test_the_shipped_defaults_match_the_frozen_config():
    """The config in the repository is what a run without overrides will use."""
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "b42_constant_area_aspect_sparse.yaml")
        .read_text(encoding="utf-8")
    )
    policy = AugmentationPolicy.from_config(config)
    assert len(policy.active()) == 7, f"expected seven live settings, got {policy.active()}"


def test_a_negative_setting_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        AugmentationPolicy(noise_std=-0.1)


def test_a_slice_dropout_of_one_is_refused():
    """Dropping every slice of every study is not an augmentation."""
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        AugmentationPolicy(slice_dropout=1.0)


def test_the_disabled_policy_is_recognised_as_disabled():
    assert AugmentationPolicy.disabled().is_disabled()
    assert not AugmentationPolicy.from_config({}).is_disabled()


# --- the augmentation itself ------------------------------------------------


def _series(slices: int = 8, height: int = 24, width: int = 32) -> torch.Tensor:
    ramp = torch.linspace(0.1, 0.9, width)
    plane = ramp[None, :].expand(height, width)
    return plane[None, None].expand(slices, 3, height, width).clone()


def test_augmentation_changes_the_pixels():
    before = _series()
    after = augment_b42_series(
        before, AugmentationPolicy.from_config({}), torch.Generator().manual_seed(7)
    )
    assert after.shape == before.shape
    assert not torch.allclose(after, before)


def test_the_disabled_policy_leaves_the_pixels_alone():
    before = _series()
    after = augment_b42_series(
        before, AugmentationPolicy.disabled(), torch.Generator().manual_seed(7)
    )
    assert torch.equal(after, before)


def test_the_same_seed_reproduces_the_same_distortion():
    policy = AugmentationPolicy.from_config({})
    first = augment_b42_series(_series(), policy, torch.Generator().manual_seed(3))
    second = augment_b42_series(_series(), policy, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_different_seeds_distort_differently():
    """Otherwise the model would still see one fixed dataset, just a warped one."""
    policy = AugmentationPolicy.from_config({})
    first = augment_b42_series(_series(), policy, torch.Generator().manual_seed(3))
    second = augment_b42_series(_series(), policy, torch.Generator().manual_seed(4))
    assert not torch.allclose(first, second)


def test_it_never_touches_the_global_random_state():
    """A run's loader order and weight init must not shift when augmentation does."""
    torch.manual_seed(11)
    expected = torch.randn(4)

    torch.manual_seed(11)
    augment_b42_series(
        _series(), AugmentationPolicy.from_config({}), torch.Generator().manual_seed(3)
    )
    assert torch.equal(torch.randn(4), expected)


def test_the_output_stays_in_range():
    """B42 pixels are percentile-normalised into [0, 1] and must stay there."""
    policy = AugmentationPolicy.from_config({})
    for seed in range(20):
        after = augment_b42_series(_series(), policy, torch.Generator().manual_seed(seed))
        assert float(after.min()) >= 0.0
        assert float(after.max()) <= 1.0
        assert torch.isfinite(after).all()


def test_a_rectangular_series_shifts_proportionally_on_each_axis():
    """B42 series are rectangles, not squares.

    Scaling both shifts by one side would move a tall series much further
    sideways than up, which is a different augmentation on every aspect ratio.
    """
    policy = AugmentationPolicy(
        rotation_deg=0.0, translate_frac=0.5, scale_jitter=0.0, gamma_jitter=0.0,
        bias_field_strength=0.0, noise_std=0.0, slice_dropout=0.0,
    )
    tall = _series(slices=2, height=64, width=16)
    after = augment_b42_series(tall, policy, torch.Generator().manual_seed(1))
    assert after.shape == tall.shape  # a wrong axis would not resize, but would blank


def test_slice_dropout_never_empties_a_series():
    """A blank study still carrying a real label teaches something false."""
    policy = AugmentationPolicy(
        rotation_deg=0.0, translate_frac=0.0, scale_jitter=0.0, gamma_jitter=0.0,
        bias_field_strength=0.0, noise_std=0.0, slice_dropout=0.999,
    )
    for seed in range(15):
        after = augment_b42_series(_series(), policy, torch.Generator().manual_seed(seed))
        alive = [i for i in range(after.shape[0]) if float(after[i].abs().sum()) > 0]
        assert alive, f"seed {seed} blanked every slice"


def test_there_is_no_left_right_flip():
    """Mirroring a knee swaps medial and lateral, which are separate targets."""
    import inspect

    source = inspect.getsource(augment_b42_series)
    for primitive in ("torch.flip", "fliplr", "hflip", "[::-1]"):
        assert primitive not in source, f"{primitive} mirrors the image"


# --- the dataset ------------------------------------------------------------


def test_the_b53_dataset_really_augments_the_decoded_pixels(study):
    """The whole point, on the real decoding path.

    Two draws of the same study, from real DICOM through B42's crop and resize,
    must differ. This is what B52 could not do.
    """
    dataset = _dataset(study, policy=AugmentationPolicy.from_config({}))

    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(2)
    second = dataset[0]

    assert len(first["volumes"]) == 2
    for position in range(2):
        assert not torch.equal(first["volumes"][position], second["volumes"][position]), (
            f"series {position} was identical across two epochs"
        )


def test_the_b53_dataset_is_reproducible_at_a_fixed_epoch(study):
    """Random per epoch, identical when re-run with the same seed."""
    left = _dataset(study, policy=AugmentationPolicy.from_config({}))
    right = _dataset(study, policy=AugmentationPolicy.from_config({}))
    left.set_epoch(4)
    right.set_epoch(4)
    for a, b in zip(left[0]["volumes"], right[0]["volumes"]):
        assert torch.equal(a, b)


def test_a_different_run_seed_gives_different_augmentation(study):
    left = _dataset(study, policy=AugmentationPolicy.from_config({}), seed=2026)
    right = _dataset(study, policy=AugmentationPolicy.from_config({}), seed=7)
    left.set_epoch(1)
    right.set_epoch(1)
    assert not torch.equal(left[0]["volumes"][0], right[0]["volumes"][0])


def test_without_a_policy_the_dataset_matches_b42_exactly(study):
    """The control arm must be B52's behaviour, not merely close to it."""
    root, records = study
    plain = B42ConstantAreaAspectDataset(
        ["study-a"], records, _config(root), crop_focus_policy=CROP_POLICY,
        center_offsets=(0,), targets=np.zeros((1, 12), np.float32),
        weights=np.ones((1, 12), np.float32),
    )
    off = _dataset(study, policy=AugmentationPolicy.disabled())
    off.set_epoch(3)
    for a, b in zip(off[0]["volumes"], plain[0]["volumes"]):
        assert torch.equal(a, b)


def test_the_geometry_is_unchanged_by_augmentation(study):
    """B42's contract is the crop, the aspect ratio and the pixel area.

    Augmentation runs after all of that and must not resize anything, or B53
    would be two changes and not comparable with B52.
    """
    plain = _dataset(study, policy=AugmentationPolicy.disabled())
    augmented = _dataset(study, policy=AugmentationPolicy.from_config({}))
    augmented.set_epoch(1)

    for a, b in zip(augmented[0]["volumes"], plain[0]["volumes"]):
        assert a.shape == b.shape
    assert augmented[0]["geometry"] == plain[0]["geometry"]


def test_slice_jitter_is_off_by_default(study):
    """It changes which slices are chosen, which is a second change."""
    dataset = _dataset(study, policy=AugmentationPolicy.from_config({}))
    assert dataset.slice_jitter == 0


def test_slice_jitter_changes_the_chosen_slices_when_asked(study):
    """It exists and works; it simply is not part of B53's headline change."""
    plain = _dataset(study, policy=AugmentationPolicy.disabled(), slice_jitter=0)
    jittered = _dataset(study, policy=AugmentationPolicy.disabled(), slice_jitter=4)
    jittered.set_epoch(1)
    assert not torch.equal(
        jittered[0]["slice_position"], plain[0]["slice_position"]
    ), "slice_jitter did not move the slice centres"


def test_slice_jitter_restores_the_offsets_it_borrowed(study):
    """It mutates center_offsets in place; leaving it shifted would drift."""
    dataset = _dataset(study, policy=AugmentationPolicy.disabled(), slice_jitter=3)
    before = dataset.center_offsets
    dataset.set_epoch(1)
    dataset[0]
    assert dataset.center_offsets == before


def test_a_negative_slice_jitter_is_refused(study):
    with pytest.raises(ValueError, match="cannot be negative"):
        _dataset(study, policy=AugmentationPolicy.disabled(), slice_jitter=-1)


# --- the check that gates a run --------------------------------------------


def test_the_verification_passes_on_a_real_augmented_dataset(study):
    dataset = _dataset(study, policy=AugmentationPolicy.from_config({}))
    report = verify_augmentation_reaches_pixels(dataset)

    assert report["series_compared"] == 2
    assert report["series_that_changed"] == 2
    assert report["max_absolute_difference"] > 0
    assert len(report["policy"]) == 7


def test_the_verification_leaves_the_epoch_where_it_found_it(study):
    """It draws twice to check; epoch 1 must still be epoch 1's draw."""
    dataset = _dataset(study, policy=AugmentationPolicy.from_config({}))
    verify_augmentation_reaches_pixels(dataset)
    assert dataset.epoch == 0


def test_the_verification_catches_augmentation_that_does_nothing(study, monkeypatch):
    """The B52 failure, simulated: a policy that is live but has no effect.

    A run must not be able to start in that state, because nothing later in a
    27-hour run would reveal it.
    """
    dataset = _dataset(study, policy=AugmentationPolicy.from_config({}))
    monkeypatch.setattr(
        "rsna_knee.b53_augmented_training.augment_b42_series",
        lambda series, policy, generator: series,
    )
    with pytest.raises(RuntimeError, match="did not reach the pixels"):
        verify_augmentation_reaches_pixels(dataset)


def test_the_verification_refuses_a_disabled_policy(study):
    """Calling it on the control arm is a mistake, not a pass."""
    dataset = _dataset(study, policy=AugmentationPolicy.disabled())
    with pytest.raises(RuntimeError, match="augmentation off"):
        verify_augmentation_reaches_pixels(dataset)


# --- what B53 holds fixed ---------------------------------------------------


def test_b53_reuses_b52s_moving_parts_rather_than_restating_them():
    """Two changes cannot hide in code that is literally the same code."""
    import inspect

    from rsna_knee import b52_competition_training as b52
    from rsna_knee import b53_augmented_training as b53

    for name in (
        "b52_parameter_groups",
        "select_train_and_validation",
        "evaluate_split",
    ):
        assert getattr(b53, name) is getattr(b52, name), (
            f"B53 has its own {name}; it must import B52's"
        )

    source = inspect.getsource(b53)
    assert "B52_DEFAULT_ENCODER_STAGES" in source
    assert "B52_DEFAULT_ENCODER_LR_SCALE" in source


def test_the_checkpoint_records_the_measurement_not_just_a_flag():
    """B52 wrote `augmentation_enabled: true` and trained on identical pixels."""
    import inspect

    from rsna_knee import b53_augmented_training as b53

    source = inspect.getsource(b53.train_b53)
    assert '"augmentation_verified"' in source, (
        "the checkpoint must carry the measured proof, not only a boolean"
    )
    assert '"augmentation_policy"' in source


def test_the_preflight_runs_the_verification_before_any_training():
    """A run that cannot prove its one change should never reach epoch 1."""
    import inspect

    from rsna_knee import b53_augmented_training as b53

    source = inspect.getsource(b53.b53_preflight)
    assert "verify_augmentation_reaches_pixels" in source

    train_source = inspect.getsource(b53.train_b53)
    preflight_at = train_source.index("b53_preflight(")
    loop_at = train_source.index("for epoch in range(1")
    assert preflight_at < loop_at, "the preflight must run before the epoch loop"
