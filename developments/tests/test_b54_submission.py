"""Submitting B54 through B42's dual-T4 path.

B54's checkpoint declares itself as B52, so nearly all of `require_b52_endpoint`
already applies to it. Exactly one thing breaks -- the strict load of a
`base_state` carrying `spacing_conditioning.projection.weight` -- and exactly
one thing is added: the submitted forward runs with the conditioning switched
off, because B42's inference loop supplies no spacing.

The tests that earn their place are the ones guarding a silent failure:
strictness must stay (relaxing it discards the trained term and still scores),
and the conditioning must really be off on the loaded model rather than off in
the argument that was passed.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna_knee.b52_competition_submission_dualgpu_fast import (
    b52_endpoint_manifest,
    require_b52_endpoint,
)
from rsna_knee.b54_competition_submission_dualgpu_fast import (
    B52_LEADERBOARD_SCORE,
    B52_LEADERBOARD_TRAINING_STUDIES,
    B54_SUBMISSION_EXPERIMENT,
    assert_conditioning_disabled,
    b54_endpoint_manifest,
    conditioning_spread_ratio,
    load_ablation,
    load_b54_checkpoint_for_submission,
    require_b54_endpoint,
    require_disabled_arm_is_safe,
)
from rsna_knee.b54_spacing_run import install_spacing_conditioning
from rsna_knee.spacing_conditioning import SPACING_BASIS, SpacingConditioning
from rsna_knee.spacing_conditioning_probe import SPREAD_PRESENT

D_MODEL = 8


class _Base(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _Base()


def _payload(**overrides) -> dict:
    """A checkpoint that passes every B52 check, so B54's own can be isolated."""
    payload = {
        "experiment": "B52_COMPETITION_FULL_FINETUNE",
        "base_state": {"spacing_conditioning.projection.weight": torch.zeros(4, 8)},
        "head_state": {"gate": torch.zeros(1)},
        "model_state": {"grid_size": 6, "top_k": 8, "temperature": 1.0},
        "spacing": {
            "enabled": True,
            "conditioning_moved": True,
            "conditioning_scale": 90.334,
            "conditioning_sites": 1,
        },
        "training_studies": 1447,
        "selected_epoch": 4,
        "selection_value": 0.804168,
    }
    payload.update(overrides)
    return payload


class _StubIdentity(dict):
    pass


@pytest.fixture
def stub_b52(monkeypatch):
    """Stand in for `require_b52_endpoint` so B54's own checks can be tested.

    Reproducing a checkpoint complete enough to satisfy every B52 assertion
    would test B52's loader, which has its own suite, and would hide which
    assertion a failure came from.
    """
    calls = []

    def fake(payload):
        calls.append(payload)
        return {
            "sparse_mil": {"grid_size": 6, "top_k": 8, "temperature": 1.0},
            "encoder_finetune": {"encoder_trainable_stages": 5},
            "encoder_chunk_size": 4,
            "selected_epoch": payload.get("selected_epoch"),
            "selection_value": payload.get("selection_value"),
            "training_studies": payload.get("training_studies"),
            "validation_studies": 548,
            "augmentation_enabled": True,
            "seed": 2026,
        }

    monkeypatch.setattr(
        "rsna_knee.b54_competition_submission_dualgpu_fast.require_b52_endpoint",
        fake,
    )
    return calls


# --- B52's checks are called, not reproduced ----------------------------------


def test_every_b52_check_still_runs(stub_b52):
    """B54 adds checks; it must not replace any."""
    require_b54_endpoint(_payload())
    assert len(stub_b52) == 1


def test_the_b52_identity_is_carried_through(stub_b52):
    identity = require_b54_endpoint(_payload())
    assert identity["encoder_chunk_size"] == 4
    assert identity["sparse_mil"]["grid_size"] == 6


def test_it_does_not_copy_b52s_assertions():
    """A reproduction would drift; a call cannot."""
    source = inspect.getsource(require_b54_endpoint)
    assert "require_b52_endpoint(payload)" in source
    assert "gold_labels_used" not in source
    assert "encoder_chunk_size" not in source.split("identity =")[0]


# --- the two checks that are only meaningful for B54 --------------------------


def test_a_checkpoint_trained_without_spacing_is_refused(stub_b52):
    """It is a B52 run, and has its own launcher."""
    with pytest.raises(ValueError, match="not trained with the spacing"):
        require_b54_endpoint(_payload(spacing={"enabled": False}))


def test_a_checkpoint_with_no_spacing_block_at_all_is_refused(stub_b52):
    payload = _payload()
    del payload["spacing"]
    with pytest.raises(ValueError, match="not trained with the spacing"):
        require_b54_endpoint(payload)


def test_a_conditioning_that_never_moved_is_refused(stub_b52):
    """Zero throughout means the weights are numerically B52's."""
    with pytest.raises(ValueError, match="never moved off zero"):
        require_b54_endpoint(
            _payload(spacing={"enabled": True, "conditioning_moved": False})
        )


def test_an_audit_that_disagrees_with_the_weights_is_refused(stub_b52):
    """The audit says trained; the state dict has no such key."""
    with pytest.raises(ValueError, match="carries no spacing_conditioning"):
        require_b54_endpoint(_payload(base_state={"something.else": torch.zeros(1)}))


def test_the_recorded_scale_reaches_the_identity(stub_b52):
    identity = require_b54_endpoint(_payload())
    assert identity["spacing"]["conditioning_scale"] == pytest.approx(90.334)


def test_a_checkpoint_with_no_recorded_scale_reads_as_one(stub_b52):
    """That is what the first B54 run trained at, so it is the honest default."""
    identity = require_b54_endpoint(
        _payload(spacing={"enabled": True, "conditioning_moved": True})
    )
    assert identity["spacing"]["conditioning_scale"] == 1.0


# --- the submitted arm is the disabled one ------------------------------------


def test_a_disabled_conditioning_passes():
    model = _Model()
    install_spacing_conditioning(model.base, enabled=False)
    assert assert_conditioning_disabled(model) == 1


def test_an_enabled_conditioning_is_refused():
    """One keyword away from its opposite, and the failure would be silent."""
    model = _Model()
    install_spacing_conditioning(model.base, enabled=True)
    with pytest.raises(RuntimeError, match="are enabled"):
        assert_conditioning_disabled(model)


def test_a_model_with_no_conditioning_is_refused():
    """It would mean the trained term was dropped rather than loaded."""
    with pytest.raises(RuntimeError, match="no SpacingConditioning"):
        assert_conditioning_disabled(_Model())


def test_a_disabled_module_really_contributes_zero():
    """The reason switching it off is enough, checked rather than asserted."""
    module = SpacingConditioning(D_MODEL, enabled=False)
    with torch.no_grad():
        module.projection.weight.normal_()

    out = module(torch.tensor([0.8, 3.3, 5.0]))
    assert torch.equal(out, torch.zeros(3, D_MODEL))


# --- the gate: may this checkpoint be submitted with the spacing off? ---------


def _weighted_payload(column_value: float, embedding_scale: float = 1.0, **overrides):
    """A payload whose conditioning weight and metadata scale are both chosen.

    Feature 0 of the basis is the normalised log spacing, so putting mass there
    produces a contribution that really does move across the corpus range --
    the quantity the gate measures.
    """
    torch.manual_seed(0)
    weight = torch.zeros(D_MODEL, SPACING_BASIS)
    weight[:, 0] = column_value

    base_state = {"spacing_conditioning.projection.weight": weight}
    for name, rows in (
        ("plane_embedding.weight", 4),
        ("fluid_embedding.weight", 3),
        ("fat_embedding.weight", 3),
    ):
        values = torch.randn(rows, D_MODEL) * embedding_scale
        values[0] = 0.0
        base_state[name] = values
    return _payload(base_state=base_state, **overrides)


def _ablation(tmp_path, checkpoint, *, delta: float, resolution: float = 0.03):
    path = tmp_path / "expert58.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "spacing_delta": delta,
                "resolution": resolution,
                "arms": {
                    "spacing_on": {"macro_auc": 0.68 + delta},
                    "spacing_off": {"macro_auc": 0.68},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_ratio_agrees_with_the_probe():
    """The gate must not be able to disagree with the tool people read."""
    from rsna_knee.spacing_conditioning_probe import probe

    payload = _weighted_payload(0.05)
    path = Path(__import__("tempfile").mkdtemp()) / "c.pt"
    torch.save(payload, path)

    assert conditioning_spread_ratio(payload) == pytest.approx(
        probe(path)["spread_over_metadata"], rel=1e-6
    )


def test_the_recorded_scale_is_applied_to_the_ratio():
    """v1 trained at 1.0 and v2 at 90.3; the same weight means different things."""
    small = _weighted_payload(0.001)
    small["spacing"] = {**small["spacing"], "conditioning_scale": 1.0}
    large = _weighted_payload(0.001)
    large["spacing"] = {**large["spacing"], "conditioning_scale": 100.0}

    assert conditioning_spread_ratio(large) == pytest.approx(
        conditioning_spread_ratio(small) * 100.0, rel=1e-5
    )


def test_a_payload_with_no_recorded_scale_reads_at_one():
    """What B54 v1 trained at, so its ratio stays reproducible."""
    payload = _weighted_payload(0.001)
    payload["spacing"] = {"enabled": True, "conditioning_moved": True}
    explicit = _weighted_payload(0.001)
    explicit["spacing"] = {**explicit["spacing"], "conditioning_scale": 1.0}

    assert conditioning_spread_ratio(payload) == pytest.approx(
        conditioning_spread_ratio(explicit)
    )


def test_a_negligible_term_needs_no_evidence():
    """B54 v1's case: 0.07% of its own sum, so switching it off changes nothing."""
    payload = _weighted_payload(1e-6)
    evidence = require_disabled_arm_is_safe(payload, checkpoint="x.pt", ablation=None)

    assert evidence["term_is_negligible"] is True
    assert evidence["spread_over_metadata"] < SPREAD_PRESENT
    assert "not a change to the model" in evidence["why_disabling_is_safe"]


def test_a_real_term_without_evidence_is_refused():
    """B54 v2's case, and the bug this whole change exists to prevent."""
    payload = _weighted_payload(1.0)
    assert conditioning_spread_ratio(payload) >= SPREAD_PRESENT

    with pytest.raises(ValueError, match="needs evidence"):
        require_disabled_arm_is_safe(payload, checkpoint="x.pt", ablation=None)


def test_a_real_term_with_an_agreeing_ablation_is_allowed(tmp_path):
    payload = _weighted_payload(1.0)
    ablation = _ablation(tmp_path, "x.pt", delta=0.0001)

    evidence = require_disabled_arm_is_safe(
        payload, checkpoint="x.pt", ablation=ablation
    )
    assert evidence["term_is_negligible"] is False
    assert evidence["expert58_spacing_delta"] == pytest.approx(0.0001)
    assert evidence["ablation_source"] == str(ablation.resolve())


def test_a_real_term_whose_ablation_moved_is_refused(tmp_path):
    """The term is doing work, so the disabled arm is not the trained model."""
    payload = _weighted_payload(1.0)
    ablation = _ablation(tmp_path, "x.pt", delta=0.05)

    with pytest.raises(ValueError, match="measurable work"):
        require_disabled_arm_is_safe(payload, checkpoint="x.pt", ablation=ablation)


def test_a_negative_ablation_delta_is_judged_on_its_size(tmp_path):
    """A term that helps and one that hurts are both measurable work."""
    payload = _weighted_payload(1.0)
    ablation = _ablation(tmp_path, "x.pt", delta=-0.05)

    with pytest.raises(ValueError, match="measurable work"):
        require_disabled_arm_is_safe(payload, checkpoint="x.pt", ablation=ablation)


def test_an_ablation_of_another_checkpoint_is_refused(tmp_path):
    """Evidence from a sibling run prints identically and licenses nothing."""
    payload = _weighted_payload(1.0)
    ablation = _ablation(tmp_path, "other.pt", delta=0.0001)

    with pytest.raises(ValueError, match="cannot license this one"):
        require_disabled_arm_is_safe(payload, checkpoint="x.pt", ablation=ablation)


def test_an_ablation_with_no_recorded_checkpoint_is_refused(tmp_path):
    path = tmp_path / "expert58.json"
    path.write_text(json.dumps({"spacing_delta": 0.0, "resolution": 0.03}), "utf-8")

    with pytest.raises(ValueError, match="cannot license this one"):
        load_ablation(path, checkpoint="x.pt")


def test_a_file_that_is_not_an_ablation_is_refused(tmp_path):
    path = tmp_path / "expert58.json"
    path.write_text(
        json.dumps({"checkpoint": str(Path("x.pt").resolve()), "spacing_delta": 0.0}),
        "utf-8",
    )

    with pytest.raises(ValueError, match="missing resolution"):
        load_ablation(path, checkpoint="x.pt")


def test_the_ablations_own_resolution_is_used(tmp_path):
    """Not a constant here: the eval owns what its surface can resolve."""
    payload = _weighted_payload(1.0)
    generous = _ablation(tmp_path, "x.pt", delta=0.02, resolution=0.5)

    evidence = require_disabled_arm_is_safe(
        payload, checkpoint="x.pt", ablation=generous
    )
    assert evidence["expert58_resolution"] == 0.5


# --- the load, and the one line that differs from B52's -----------------------


def _body(function) -> str:
    source = inspect.getsource(function).lstrip()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    statements = node.body[1:] if ast.get_docstring(node) else node.body
    return "\n".join(source.splitlines()[statements[0].lineno - 1 :])


def test_the_conditioning_is_installed_before_the_load():
    body = _body(load_b54_checkpoint_for_submission)
    assert body.index("install_spacing_conditioning") < body.index("load_state_dict")


def test_the_load_stays_strict():
    """`strict=False` would succeed by discarding the trained term.

    Read out of the syntax tree rather than searched for as text: this
    function's own comments discuss `strict=False` to explain why it is not
    used, and a string search cannot tell prose from a call.
    """
    tree = ast.parse(inspect.getsource(load_b54_checkpoint_for_submission).lstrip())
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    ]

    assert len(loads) == 2, "the base and the head are both loaded"
    for call in loads:
        strict = {kw.arg: kw.value for kw in call.keywords}.get("strict")
        assert isinstance(strict, ast.Constant) and strict.value is True


def test_it_installs_the_conditioning_disabled():
    assert "enabled=False" in _body(load_b54_checkpoint_for_submission)


def test_the_loaded_model_is_checked_not_the_argument():
    assert "assert_conditioning_disabled(model)" in _body(
        load_b54_checkpoint_for_submission
    )


def test_the_encoder_fingerprint_is_still_verified():
    body = _body(load_b54_checkpoint_for_submission)
    assert "encoder_state_sha256" in body
    assert "fingerprint changed" in body


def test_the_base_checkpoint_fingerprint_is_still_verified():
    assert "base checkpoint fingerprint mismatch" in _body(
        load_b54_checkpoint_for_submission
    )


def test_installing_first_is_what_makes_the_key_loadable():
    """The concrete failure the ordering exists to prevent."""
    trained = _Base()
    install_spacing_conditioning(trained)
    with torch.no_grad():
        trained.spacing_conditioning.projection.weight.normal_()
    state = trained.state_dict()

    with pytest.raises(RuntimeError, match="Unexpected"):
        _Base().load_state_dict(state, strict=True)

    loaded = _Base()
    install_spacing_conditioning(loaded, enabled=False)
    loaded.load_state_dict(state, strict=True)
    assert torch.allclose(
        loaded.spacing_conditioning.projection.weight,
        trained.spacing_conditioning.projection.weight,
    )
    assert loaded.spacing_conditioning.enabled is False


def test_the_recorded_scale_survives_a_reload():
    """It is not in the state dict, so only the install can carry it."""
    loaded = _Base()
    install_spacing_conditioning(loaded, enabled=False, scale=90.334)
    assert loaded.spacing_conditioning.scale == pytest.approx(90.334)


def test_the_projection_width_matches_the_basis():
    loaded = _Base()
    conditioning = install_spacing_conditioning(loaded, enabled=False)
    assert conditioning.projection.weight.shape == (D_MODEL, SPACING_BASIS)


# --- the manifest -------------------------------------------------------------


def _manifest_payload() -> dict:
    return _payload(
        version="b52_competition_full_finetune_v1",
        history=[{}, {}, {}, {}, {}, {}],
        selection_metric="macro_auc",
        epochs_planned=6,
        validation_studies=548,
        train_splits=["a"],
        augmentation_enabled=True,
        encoder_trainable_stages=5,
        changed_from_frozen_contract=["epochs"],
    )


def test_the_manifest_states_b54s_identity():
    manifest = b54_endpoint_manifest(_manifest_payload())
    assert manifest["experiment"] == B54_SUBMISSION_EXPERIMENT
    assert manifest["version"].lower().startswith("b54")
    # B52's own version is kept, so the manifest still says which inference
    # contract the run was executed under.
    assert manifest["b52_manifest_version"] == "b52_competition_full_finetune_v1"


def test_b52s_manifest_is_reused_not_rewritten():
    """Everything about the training contract applies to B54 unchanged."""
    payload = _manifest_payload()
    b52 = b52_endpoint_manifest(payload)
    b54 = b54_endpoint_manifest(payload)

    assert b54["fixed_endpoint"] == b52["fixed_endpoint"] is False
    assert b54["selected_epoch"] == b52["selected_epoch"]
    assert b54["governance"] == b52["governance"]


def test_b52s_own_manifest_is_untouched():
    """B54 must not change what a B52 submission claims about itself."""
    payload = _manifest_payload()
    before = dict(b52_endpoint_manifest(payload))
    b54_endpoint_manifest(payload)
    assert b52_endpoint_manifest(payload) == before


def test_the_manifest_says_which_arm_was_submitted():
    spacing = b54_endpoint_manifest(_manifest_payload())["spacing_conditioning"]

    assert spacing["trained"] is True
    assert spacing["enabled_at_inference"] is False
    assert spacing["submitted_arm"] == "spacing_off"


def test_the_manifest_carries_the_measured_evidence_it_was_given():
    """A claim that the term is a no-op needs its measurement beside it."""
    evidence = {"spread_over_metadata": 0.0007, "term_is_negligible": True}
    spacing = b54_endpoint_manifest(
        _manifest_payload(), spacing_evidence=evidence
    )["spacing_conditioning"]

    assert spacing["evidence"] == evidence


def test_the_manifest_quotes_no_hardcoded_expert58_numbers():
    """The bug this replaced: v1's numbers attached to v2's weights.

    With no evidence supplied the block must say so, never fall back on a
    constant that describes some other run.
    """
    spacing = b54_endpoint_manifest(_manifest_payload())["spacing_conditioning"]

    assert spacing["evidence"] is None
    text = json.dumps(spacing)
    for stale in ("0.681223", "0.681071", "0.000152"):
        assert stale not in text


def test_the_manifest_warns_that_the_population_differs():
    warning = b54_endpoint_manifest(_manifest_payload())["population_warning"]

    assert warning["b54_training_studies"] == 1447
    assert warning["leaderboard_reference_training_studies"] == 3801
    assert warning["comparable"] is False


def test_a_full_population_checkpoint_reads_as_comparable():
    """The warning must not fire on a run that really is comparable."""
    payload = _manifest_payload()
    payload["training_studies"] = B52_LEADERBOARD_TRAINING_STUDIES
    warning = b54_endpoint_manifest(payload)["population_warning"]

    assert warning["comparable"] is True


def test_the_leaderboard_reference_is_the_measured_one():
    assert B52_LEADERBOARD_SCORE == 0.716
    assert B52_LEADERBOARD_TRAINING_STUDIES == 3801


def test_the_manifest_names_what_b54_actually_changed():
    changes = b54_endpoint_manifest(_manifest_payload())["what_b54_changes_from_b52"]
    joined = " ".join(changes).lower()

    assert "teacher" in joined
    assert "v1.3.1" in joined
    assert "disabled" in joined


# --- the launcher -------------------------------------------------------------


def test_the_launcher_hands_b42_b54s_loader_and_manifest():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    body = _body(generate_b54_submission_dual_gpu_fast)
    assert "load_replica=_load_b54_replica" in body
    assert "b54_endpoint_manifest(p, spacing_evidence=evidence)" in body


def test_the_launcher_gates_before_it_spends_a_gpu():
    """Eight hours in is the wrong place to discover the arm is not licensed."""
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    body = _body(generate_b54_submission_dual_gpu_fast)
    assert body.index("require_disabled_arm_is_safe") < body.index(
        "generate_b42_submission_dual_gpu_fast"
    )


def test_the_launcher_accepts_the_ablation():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    parameters = inspect.signature(generate_b54_submission_dual_gpu_fast).parameters
    assert parameters["expert58_ablation"].default is None


def test_the_launcher_keeps_b52s_hidden_safe_defaults():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    parameters = inspect.signature(generate_b54_submission_dual_gpu_fast).parameters
    assert parameters["stream_views"].default is True
    assert parameters["abort_on_budget"].default is False
    assert parameters["on_unreadable"].default == "fallback"


def test_the_launcher_requires_a_declared_hash():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    parameters = inspect.signature(generate_b54_submission_dual_gpu_fast).parameters
    assert parameters["expected_checkpoint_sha256"].default is inspect.Parameter.empty


def test_the_launcher_announces_the_population_gap():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    body = _body(generate_b54_submission_dual_gpu_fast)
    assert "POPULATION" in body
    assert "not " in body


def test_the_launcher_announces_the_disabled_conditioning():
    from rsna_knee.b54_competition_submission_dualgpu_fast import (
        generate_b54_submission_dual_gpu_fast,
    )

    assert "DISABLED" in _body(generate_b54_submission_dual_gpu_fast)
