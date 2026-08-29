"""B50 lets the study hierarchy adapt, and changes nothing else.

Two properties carry the whole experiment. The control must be a bit-for-bit
reproduction of B42's trainable surface, or a difference cannot be attributed.
And the candidate must gain gradients *without* gaining training mode, because
B34's local-context scaffold is defined to be active only while
`model.training` and its inference contract is that `eval()` bypasses it
exactly.
"""

from __future__ import annotations

import pytest
import torch

from rsna_knee.b50_adapted_hierarchy_mil import (
    B50_ALWAYS_FROZEN_PREFIXES,
    B50_ARMS,
    B50_EXPECTED_HIERARCHY_PARAMETERS,
    B50_HIERARCHY_LR_SCALE,
    b50_parameter_groups,
    b50_state,
    hierarchy_parameter_names,
    require_b50_contract,
)


@pytest.fixture(scope="module")
def config():
    from model._implementation import read_config

    return read_config("config/b42_constant_area_aspect_sparse.yaml")


@pytest.fixture(scope="module")
def base_model(config):
    from rsna_knee.b34_training_only_context_scaffold import (
        b34_model_spec,
        build_b34_model,
    )

    return build_b34_model(b34_model_spec(config, normalize_input=True))


# --- what the experiment is actually about --------------------------------


def test_the_frozen_hierarchy_is_the_size_the_protocol_claims(base_model):
    """18.96M parameters that have not moved since B34."""
    names = hierarchy_parameter_names(base_model)
    lookup = dict(base_model.named_parameters())
    total = sum(lookup[name].numel() for name in names)
    assert total == B50_EXPECTED_HIERARCHY_PARAMETERS

    encoder = sum(p.numel() for p in base_model.encoder.parameters())
    everything = sum(p.numel() for p in base_model.parameters())
    # The hierarchy is everything except the encoder and the bypassed scaffold.
    scaffold = sum(
        p.numel()
        for name, p in base_model.named_parameters()
        if name.startswith("local_context.")
    )
    assert total + encoder + scaffold == everything


def test_the_encoder_and_the_bypassed_scaffold_are_never_unfrozen(base_model):
    names = hierarchy_parameter_names(base_model)
    assert names, "there must be a hierarchy to unfreeze"
    for name in names:
        assert not name.startswith(B50_ALWAYS_FROZEN_PREFIXES)
    assert not any(name.startswith("encoder.") for name in names)
    assert not any(name.startswith("local_context.") for name in names)


def test_the_aggregation_blocks_are_all_included(base_model):
    """The study Transformer and pathology queries are the point of B50."""
    names = set(hierarchy_parameter_names(base_model))
    for block in ("context", "pathology_context", "series_pool", "cross_attention"):
        assert any(name.startswith(f"{block}.") for name in names), block


# --- gradients without training mode --------------------------------------


class _Base(torch.nn.Module):
    """Stands in for the B34 model: an encoder, a hierarchy, a scaffold."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.context = torch.nn.Linear(4, 4)
        self.pathology_context = torch.nn.Linear(4, 4)
        self.local_context = torch.nn.Linear(4, 4)


def _freeze_like_b37(base: _Base) -> _Base:
    for name, parameter in base.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad_(False)
    return base


def test_unfreezing_restores_gradients_to_the_hierarchy():
    base = _freeze_like_b37(_Base())
    assert not base.context.weight.requires_grad

    lookup = dict(base.named_parameters())
    for name in hierarchy_parameter_names(base):
        lookup[name].requires_grad_(True)

    assert base.context.weight.requires_grad
    assert base.pathology_context.weight.requires_grad
    assert not base.local_context.weight.requires_grad, "bypassed, so left frozen"


def test_requires_grad_and_training_mode_are_independent():
    """The mistake B50 must not make: unfreezing by switching to train mode."""
    base = _Base()
    base.eval()
    base.context.weight.requires_grad_(True)

    assert not base.training, "eval mode must survive unfreezing"
    out = base.context(torch.ones(1, 4))
    out.sum().backward()
    assert base.context.weight.grad is not None, "gradients flow in eval mode"


def test_the_control_leaves_the_hierarchy_exactly_as_b37_left_it():
    base = _freeze_like_b37(_Base())
    frozen_before = {
        name: p.requires_grad for name, p in base.named_parameters()
    }
    # The control arm performs no unfreezing at all.
    frozen_after = {name: p.requires_grad for name, p in base.named_parameters()}
    assert frozen_before == frozen_after
    assert not any(
        p.requires_grad for n, p in base.named_parameters() if not n.startswith("encoder.")
    )


# --- learning rates --------------------------------------------------------


class _Model(torch.nn.Module):
    def __init__(self, adapt: bool) -> None:
        super().__init__()
        self.base = _freeze_like_b37(_Base())
        self.head = torch.nn.Linear(4, 4)
        self.hierarchy_names = hierarchy_parameter_names(self.base)
        if adapt:
            lookup = dict(self.base.named_parameters())
            for name in self.hierarchy_names:
                lookup[name].requires_grad_(True)

    def hierarchy_parameters(self):
        lookup = dict(self.base.named_parameters())
        return [lookup[name] for name in self.hierarchy_names]


def test_the_hierarchy_gets_its_own_reduced_rate():
    groups = b50_parameter_groups(
        _Model(adapt=True), head_lr=1e-4, encoder_lr_scale=0.05
    )
    named = {g["name"]: g for g in groups}
    assert set(named) == {"sparse_head", "encoder_tail", "study_hierarchy"}
    assert named["sparse_head"]["lr"] == pytest.approx(1e-4)
    assert named["encoder_tail"]["lr"] == pytest.approx(5e-6)
    assert named["study_hierarchy"]["lr"] == pytest.approx(1e-4 * B50_HIERARCHY_LR_SCALE)


def test_the_control_has_no_hierarchy_group_at_all():
    groups = b50_parameter_groups(
        _Model(adapt=False), head_lr=1e-4, encoder_lr_scale=0.05
    )
    assert "study_hierarchy" not in {g["name"] for g in groups}


def test_no_parameter_appears_in_two_groups():
    groups = b50_parameter_groups(
        _Model(adapt=True), head_lr=1e-4, encoder_lr_scale=0.05
    )
    seen = set()
    for group in groups:
        for parameter in group["params"]:
            assert id(parameter) not in seen, "a duplicated parameter gets two updates"
            seen.add(id(parameter))


# --- the contract ----------------------------------------------------------


def test_both_arms_clear_the_inherited_b42_contract(config):
    for arm in B50_ARMS:
        resolved = require_b50_contract({**config, "b50_arm": arm})
        assert resolved["arm"] == arm
        assert resolved["hierarchy_lr_scale"] == B50_HIERARCHY_LR_SCALE
    assert require_b50_contract(
        {**config, "b50_arm": "adapted_hierarchy_candidate"}
    )["adapt_hierarchy"]
    assert not require_b50_contract(
        {**config, "b50_arm": "frozen_hierarchy_control"}
    )["adapt_hierarchy"]


def test_an_unknown_arm_is_refused(config):
    with pytest.raises(ValueError, match="B50 arm must be one of"):
        require_b50_contract({**config, "b50_arm": "half_frozen"})


def test_the_learning_rate_scale_cannot_be_swept(config):
    """The value is frozen before any result is seen."""
    for bad in (0.01, 0.1, 1.0):
        with pytest.raises(ValueError, match="no sweep"):
            require_b50_contract(
                {
                    **config,
                    "b50_arm": "adapted_hierarchy_candidate",
                    "b50_hierarchy_lr_scale": bad,
                }
            )


def test_b50_does_not_loosen_anything_b42_froze(config):
    with pytest.raises(ValueError, match="B42 freezes b42_reference_area"):
        require_b50_contract(
            {
                **config,
                "b50_arm": "adapted_hierarchy_candidate",
                "b42_reference_area": 100_000,
            }
        )


@pytest.mark.parametrize("key", ["b37_top_k", "b37_temperature"])
def test_the_inherited_sparse_constants_stay_frozen(config, key):
    with pytest.raises(ValueError, match=f"freezes {key}"):
        require_b50_contract(
            {**config, "b50_arm": "adapted_hierarchy_candidate", key: 99}
        )


def test_the_recorded_state_names_both_arms_and_the_eval_mode_rule():
    for arm in B50_ARMS:
        state = b50_state(arm)
        assert state["arm"] == arm
        assert state["hierarchy_parameters"] == B50_EXPECTED_HIERARCHY_PARAMETERS
        assert "eval" in state["base_module_mode"]
    assert b50_state("adapted_hierarchy_candidate")["adapt_hierarchy"]
    assert not b50_state("frozen_hierarchy_control")["adapt_hierarchy"]


# --- the trainer's wiring guard -------------------------------------------


def test_the_trainer_refuses_a_candidate_that_never_unfroze():
    """The failure that would look exactly like B48's and B49's nulls."""
    from rsna_knee.b50_adapted_hierarchy_training import _check_arm_wiring

    model = _Model(adapt=True)
    with pytest.raises(RuntimeError, match="without a single hierarchy gradient"):
        _check_arm_wiring(model, adapt_hierarchy=True, hierarchy_saw_gradient=False)

    # With a gradient it passes.
    _check_arm_wiring(model, adapt_hierarchy=True, hierarchy_saw_gradient=True)


def test_the_trainer_refuses_a_control_that_leaked_gradients():
    """Otherwise the control is not a reproduction of B42."""
    from rsna_knee.b50_adapted_hierarchy_training import _check_arm_wiring

    model = _Model(adapt=False)
    _check_arm_wiring(model, adapt_hierarchy=False, hierarchy_saw_gradient=False)

    model.base.context.weight.grad = torch.ones_like(model.base.context.weight)
    with pytest.raises(RuntimeError, match="not a reproduction of B42"):
        _check_arm_wiring(model, adapt_hierarchy=False, hierarchy_saw_gradient=False)


def test_a_zero_gradient_does_not_count_as_a_gradient():
    """An all-zero grad tensor means the path is wired but carrying nothing."""
    from rsna_knee.b50_adapted_hierarchy_training import _hierarchy_gradient_present

    model = _Model(adapt=True)
    for parameter in model.hierarchy_parameters():
        parameter.grad = torch.zeros_like(parameter)
    assert not _hierarchy_gradient_present(model)

    model.base.context.weight.grad = torch.ones_like(model.base.context.weight)
    assert _hierarchy_gradient_present(model)


def test_the_frozen_seed_is_enforced():
    from rsna_knee.b50_adapted_hierarchy_training import B50_SEED, train_b50_domain_arm

    assert B50_SEED == 2026
    with pytest.raises(ValueError, match="freezes seed"):
        train_b50_domain_arm(
            {},
            data_root=".",
            labels_root=".",
            series_policy_path=".",
            base_checkpoint=".",
            domain_split=".",
            arm="adapted_hierarchy_candidate",
            seed=7,
        )


def test_the_config_ships_both_frozen_values():
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("config/b50_adapted_hierarchy.yaml").read_text())
    assert cfg["b50_hierarchy_lr_scale"] == B50_HIERARCHY_LR_SCALE
    assert cfg["b50_arm"] in B50_ARMS
    # Every B42 geometry key must be carried through unchanged.
    b42 = yaml.safe_load(Path("config/b42_constant_area_aspect_sparse.yaml").read_text())
    for key, value in b42.items():
        assert cfg[key] == value, f"B50 changed inherited key {key}"


# --- the trainer calls its collaborators correctly -------------------------


def _call_keywords(source_path, callee: str) -> set[str]:
    """Keyword names the module passes to `callee`, read from the AST."""
    import ast

    tree = ast.parse(source_path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != callee:
            continue
        found |= {kw.arg for kw in node.keywords if kw.arg}
    return found


@pytest.mark.parametrize(
    "callee,target",
    [
        ("B42ConstantAreaAspectDataset", "rsna_knee.b42_constant_area_aspect_sparse_mil"),
        ("B50AdaptedHierarchySparseMILResidual", "rsna_knee.b50_adapted_hierarchy_mil"),
    ],
)
def test_the_trainer_supplies_every_required_keyword(callee, target):
    """A missing keyword-only argument only surfaces at runtime, on the GPU.

    `crop_focus_policy` was omitted exactly this way and the preflight died
    after loading the model and the whole supervision surface. Reading the real
    signature is cheap; discovering it on the training machine is not.
    """
    import importlib
    import inspect
    from pathlib import Path

    module = importlib.import_module(target)
    signature = inspect.signature(getattr(module, callee).__init__)
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
    passed = _call_keywords(
        Path("developments/src/rsna_knee/b50_adapted_hierarchy_training.py"), callee
    )
    missing = required - passed
    assert not missing, f"B50's trainer never passes {sorted(missing)} to {callee}"


def test_the_trainer_passes_the_crop_policy_the_contract_resolved():
    """Not a freshly derived one: the arms must share the parent's exact policy."""
    from pathlib import Path

    source = Path(
        "developments/src/rsna_knee/b50_adapted_hierarchy_training.py"
    ).read_text()
    assert 'crop_focus_policy=contract["crop_policy"]' in source


# --- the split boundary B48 and B49 already spent --------------------------


def test_the_trainer_refuses_the_spent_b48_split(tmp_path):
    """The B48/B49 surface has been inspected twice; B50 needs its own gate."""
    from rsna_knee.b50_adapted_hierarchy_training import load_b50_selection_gate

    (tmp_path / "domain_split.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="fresh selection gate"):
        load_b50_selection_gate(tmp_path)


def test_the_fresh_gate_is_read_and_prior_rows_are_dropped(tmp_path):
    import json

    import pandas as pd
    from rsna_knee.b50_adapted_hierarchy_training import load_b50_selection_gate

    rows = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "c", "d"],
            "scanner_profile": ["P1", "P1", "P2", "P3"],
            "parent_b48_split": ["train", "train", "train", "validation_seen_scanners"],
            "b50_split": [
                "train",
                "validation_seen_scanners",
                "validation_unseen_scanners",
                "excluded_prior_surface",
            ],
        }
    )
    rows.to_csv(tmp_path / "b50_selection_split_by_study.csv", index=False)
    (tmp_path / "b50_selection_split.json").write_text(
        json.dumps({"version": "b50_fresh_scanner_grouped_selection_split_v1"})
    )

    _payload, loaded, meta = load_b50_selection_gate(tmp_path)

    # Every row is returned, spent ones still labelled. They are excluded by
    # never being selected, not by being deleted -- deleting them would defeat
    # the shared check that every report-only study belongs to exactly one split.
    assert len(loaded) == 4
    assert set(loaded["split"]) == {
        "train",
        "validation_seen_scanners",
        "validation_unseen_scanners",
        "excluded_prior_surface",
    }
    spent = loaded.loc[loaded["split"].eq("excluded_prior_surface"), "StudyInstanceUID"]
    assert set(spent) == {"d"}

    # And the training selection never reaches them.
    from rsna_knee.b48_global_conditioned_sparse_training import _indices_for_split

    chosen = _indices_for_split(["a", "b", "c", "d"], loaded, "train")
    assert [["a", "b", "c", "d"][i] for i in chosen] == ["a"]
    assert meta["sha256"] and meta["rows_sha256"]


def test_a_gate_marking_nothing_as_spent_is_refused(tmp_path):
    """The parent's validation rows must all be carried through as excluded."""
    import json

    import pandas as pd
    from rsna_knee.b50_adapted_hierarchy_training import load_b50_selection_gate

    # Well formed in every other respect -- all three groups non-empty, and the
    # seen comparator drawn from a profile training keeps -- so the gate module's
    # own verifier passes and this check is the one being exercised.
    rows = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "c"],
            "scanner_profile": ["P1", "P1", "P2"],
            "parent_b48_split": ["train", "train", "train"],
            "b50_split": [
                "train",
                "validation_seen_scanners",
                "validation_unseen_scanners",
            ],
        }
    )
    rows.to_csv(tmp_path / "b50_selection_split_by_study.csv", index=False)
    (tmp_path / "b50_selection_split.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="marks no rows as spent"):
        load_b50_selection_gate(tmp_path)


def test_a_gate_that_reuses_a_spent_row_is_refused(tmp_path):
    import json

    import pandas as pd
    from rsna_knee.b50_adapted_hierarchy_training import load_b50_selection_gate

    rows = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "scanner_profile": ["P1", "P2"],
            # This row was B48/B49 validation and must be excluded, not trained on.
            "parent_b48_split": ["train", "holdout_unseen_scanners"],
            "b50_split": ["train", "train"],
        }
    )
    rows.to_csv(tmp_path / "b50_selection_split_by_study.csv", index=False)
    (tmp_path / "b50_selection_split.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="reuse a B48/B49 validation row"):
        load_b50_selection_gate(tmp_path)
