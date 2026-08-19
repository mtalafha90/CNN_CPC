"""Contract tests for the clean working-model interface.

These cover the interface itself, not the training results: that the public
modules stay free of experiment names, that a checkpoint's architecture is
honoured rather than assumed, and that the supervision choice maps onto the
label surfaces it claims to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from model import _implementation
from model.architecture import TARGETS, WORKING_MODEL, describe
from model.preprocessing import CROP_POLICY

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PACKAGES = ("model", "data", "training", "validation", "testing")

# `b6`, `b12_1`, `B20`, `phase9` -- the historical experiment identifiers.
EXPERIMENT_NAME = re.compile(r"\b(?:[bB]\d{1,3}(?:_\d+)?|[pP]hase\d+)\b")


def public_modules() -> list[Path]:
    modules: list[Path] = []
    for package in PUBLIC_PACKAGES:
        for path in sorted((REPOSITORY_ROOT / package).glob("*.py")):
            if path.name != "_implementation.py":
                modules.append(path)
    return modules


def test_public_interface_names_no_experiments():
    """The adapter is the only place an experiment number may appear."""
    offenders: list[str] = []
    for path in public_modules():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = EXPERIMENT_NAME.search(line)
            if match:
                relative = path.relative_to(REPOSITORY_ROOT)
                offenders.append(f"{relative}:{number}: {match.group(0)}")
    assert not offenders, "experiment names leaked into the public interface:\n" + "\n".join(
        offenders
    )


def test_public_interface_is_non_empty():
    """Guard the guard: the scan must actually be looking at files."""
    modules = public_modules()
    assert len(modules) >= 5
    assert any(path.name == "architecture.py" for path in modules)


def test_targets_are_the_twelve_scored_findings():
    assert len(TARGETS) == 12
    assert len(set(TARGETS)) == 12
    for name in ("ACL", "Effusion", "Fracture"):
        assert name in TARGETS


def test_description_records_that_expert_labels_never_train():
    description = describe()
    assert description["expert_labels_in_gradients"] == 0
    assert description is not WORKING_MODEL, "describe() must return a copy"


def test_crop_policy_is_the_deterministic_ninety_percent_crop():
    assert CROP_POLICY["crop_fraction"] == pytest.approx(0.90)


def test_supervision_choices_map_to_label_surfaces():
    assert set(_implementation.SUPERVISION_SURFACES) == {"latin-script", "all-script"}
    assert len(set(_implementation.SUPERVISION_SURFACES.values())) == 2


def test_supervision_surfaces_are_not_described_as_english():
    """Phase 5 established the frozen parser is multilingual within Latin script.

    It matched South-Slavic, Turkish and Spanish reports; calling that surface
    English-only would contradict the finding the naming exists to convey.
    """
    for name in _implementation.SUPERVISION_SURFACES:
        assert "english" not in name.lower()


def test_unknown_supervision_is_rejected_before_training():
    with pytest.raises(ValueError, match="supervision must be one of"):
        _implementation.train_working_model(
            {},
            supervision="translated",
            latin_script_labels_root="a",
            all_script_labels_root="b",
            series_policy_path="c",
            encoder_checkpoint="d",
            out_root="e",
        )


def test_working_architecture_is_rebuildable():
    assert _implementation.WORKING_ARCHITECTURE in _implementation._ARCHITECTURE_BUILDERS


def test_unknown_architecture_is_refused_with_the_supported_list():
    with pytest.raises(ValueError, match="unsupported architecture"):
        _implementation.build_network({"architecture": "something_else_v1"})


def test_run_directory_groups_every_stage_under_one_experiment(tmp_path):
    made = {
        stage: _implementation.run_directory("experiment_1", stage, runs_root=tmp_path)
        for stage in _implementation.STAGES
    }
    for stage, path in made.items():
        assert path.is_dir()
        assert path.name == stage
        assert path.parent.name == "experiment_1"
    # One experiment is one directory, so it can be archived or removed whole.
    assert len({path.parent for path in made.values()}) == 1


def test_run_directory_rejects_an_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="stage must be one of"):
        _implementation.run_directory("experiment_1", "predict", runs_root=tmp_path)


@pytest.mark.parametrize("name", ["", "  ", "a/b", "..", "."])
def test_run_directory_rejects_paths_masquerading_as_names(name, tmp_path):
    with pytest.raises(ValueError, match="single directory name"):
        _implementation.run_directory(name, "train", runs_root=tmp_path)


def test_checkpoint_without_weights_is_refused(tmp_path):
    import torch

    path = tmp_path / "incomplete.pt"
    torch.save({"model_spec": {"architecture": _implementation.WORKING_ARCHITECTURE}}, path)
    with pytest.raises(ValueError, match="model_state"):
        _implementation.load_checkpoint(path)


def _tiny_spec():
    from model._implementation import network_spec, read_config

    spec = dict(network_spec(read_config("config/current_model.yaml")))
    spec["n_slices"] = 2
    spec["encoder_batch_size"] = 4
    return spec


def _save(spec, model, path, **extra):
    import torch

    from model._implementation import encoder_fingerprint, freeze_encoder

    freeze_encoder(model)
    torch.save(
        {
            "model_spec": spec,
            "model_state": model.state_dict(),
            "encoder_sha256_final": encoder_fingerprint(model),
            **extra,
        },
        path,
    )
    return path


@pytest.mark.parametrize("variant", ["tiny", "small"])
def test_a_dinov3_checkpoint_reloads_with_its_own_encoder(variant, tmp_path):
    """The loader must rebuild the encoder the checkpoint was trained with.

    Both encoders are 768-d but their state dicts are differently named, so
    rebuilding the wrong one fails on every encoder key.
    """
    import torch

    from model._implementation import attach_dinov3, build_network, load_checkpoint

    spec = _tiny_spec()
    model = build_network(spec, pretrained_weights=False)
    replacement = attach_dinov3(model, variant=variant, pretrained_weights=False)
    spec["encoder_source"] = "dinov3"
    spec["dinov3_variant"] = variant

    path = _save(spec, model, tmp_path / "model.pt", encoder={"variant": variant})
    reloaded, _ = load_checkpoint(path)

    assert type(reloaded.encoder).__name__ == type(replacement).__name__
    assert reloaded.encoder.out_dim == 768
    for a, b in zip(model.state_dict().values(), reloaded.state_dict().values()):
        assert torch.equal(a, b)


def test_the_variant_can_come_from_the_payload_alone(tmp_path):
    """Checkpoints written before the spec recorded the variant must still load."""
    from model._implementation import attach_dinov3, build_network, load_checkpoint

    spec = _tiny_spec()
    model = build_network(spec, pretrained_weights=False)
    attach_dinov3(model, variant="tiny", pretrained_weights=False)
    spec["encoder_source"] = "dinov3"  # deliberately no dinov3_variant

    path = _save(spec, model, tmp_path / "model.pt", encoder={"variant": "tiny"})
    reloaded, _ = load_checkpoint(path)
    assert reloaded.encoder.out_dim == 768


def test_a_report_aligned_checkpoint_still_reloads_unchanged(tmp_path):
    from model._implementation import build_network, load_checkpoint

    spec = _tiny_spec()
    model = build_network(spec, pretrained_weights=False)
    path = _save(spec, model, tmp_path / "model.pt")

    reloaded, _ = load_checkpoint(path)
    assert type(reloaded.encoder).__name__ == "ConvNeXtSliceEncoder"


def test_the_seed_can_be_varied_from_the_command_line():
    """Ensembling only pays when the members differ.

    Two runs that share a seed share their initialisation, data order and
    augmentation, so they make the same mistakes and averaging them gains
    nothing. Varying the seed is what makes an ensemble worth building.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        capture_output=True, text=True,
    )
    assert "--seed" in result.stdout


@pytest.mark.parametrize("stage", ["validate", "test"])
def test_both_scoring_stages_accept_several_models(stage):
    import subprocess
    import sys

    module = "validation.validate" if stage == "validate" else "testing.test"
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert "repeat the flag" in result.stdout
