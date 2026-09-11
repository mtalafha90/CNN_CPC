"""B57 checks data lineage, actual gradients, pairing and crash recovery."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna_knee.b57_protocol import ARMS, VERSION, digest, partition, require_same_run
from rsna_knee.b57_models import DinoSliceTransformer, build_model, parameter_groups, public_weights
from rsna_knee.b57_evaluation import align_pair, bootstrap_delta, compare


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "config/b57_clean_backbone_comparison.json").read_text())


def test_partition_excludes_validation_profile_from_old_excluded_pool():
    rows = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c", "d"],
                         "scanner_profile": ["train", "valid", "valid", "train"],
                         "b50_split": ["train", "validation_unseen_scanners",
                                       "excluded_prior_surface", "excluded_prior_surface"]})
    splits, _ = partition(["a", "b", "c", "d"], rows, ["expert"])
    assert splits["train"].tolist() == [0, 3]
    assert splits["validation"].tolist() == [1]
    assert splits["excluded_profile_overlap"].tolist() == [2]
    with pytest.raises(ValueError, match="expert"):
        partition(["a", "b", "c", "d"], rows, ["c"])
    with pytest.raises(ValueError, match="population"):
        partition(["a", "b", "d"], rows, [])
    rows.loc[0, "scanner_profile"] = None
    with pytest.raises(ValueError, match="missing"):
        partition(["a", "b", "c", "d"], rows, [])


class TinyEncoder(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.conv(x).tanh().mean((-2, -1)))


def bag():
    return ([torch.rand(4, 3, 32, 64), torch.rand(4, 3, 64, 32)],
            torch.ones(2), torch.tensor([[1, 2, 1], [2, 1, 2]]),
            torch.tensor([[0., .7, .3, 1.], [0., .7, .3, 1.]]))


def tiny_model():
    return DinoSliceTransformer(TinyEncoder(), dim=12, heads=3, dropout=0., chunk_size=2)


def test_candidate_masks_absent_nan_series_and_is_series_permutation_invariant():
    model = tiny_model().eval()
    volumes, present, meta, position = bag()
    with torch.no_grad():
        expected = model(volumes, present, meta, position).logits
        reversed_ = model(volumes[::-1], present.flip(0), meta.flip(0), position.flip(0)).logits
        absent = model(volumes + [torch.full((4, 3, 32, 32), float("nan"))],
                       torch.tensor([1., 1., 0.]), torch.cat((meta, torch.zeros(1, 3, dtype=torch.long))),
                       torch.cat((position, torch.zeros(1, 4)))).logits
    torch.testing.assert_close(expected, reversed_, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(expected, absent, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="no readable"):
        model(volumes, torch.zeros(2), meta, position)


def test_checkpointed_candidate_preserves_forward_and_encoder_gradients(config):
    torch.manual_seed(12)
    checkpointed = tiny_model().train()
    ordinary = tiny_model().train()
    ordinary.load_state_dict(checkpointed.state_dict())
    ordinary.gradient_checkpointing = False
    inputs = bag()
    for model in (checkpointed, ordinary):
        model(*inputs).logits.square().mean().backward()
        groups = parameter_groups(model, ARMS[1], config)
        for group in groups:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group["params"])
    for (n, a), (m, b) in zip(checkpointed.named_parameters(), ordinary.named_parameters()):
        assert n == m
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_real_backbones_train_and_roundtrip_offline(arm, config):
    import yaml
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    settings = yaml.safe_load((root / "config/b42_constant_area_aspect_sparse.yaml").read_text())
    model = build_model(arm, settings, config).train()
    groups = parameter_groups(model, arm, config)
    assert all(p.requires_grad for p in groups[0]["params"])
    volumes = [torch.rand(32, 3, 32, 64)]
    args = (volumes, torch.ones(1), torch.tensor([[1, 2, 1]]), torch.linspace(0, 1, 32)[None])
    out = model(*args)
    loss = out.logits.square().mean()
    if arm == ARMS[0]:
        loss = loss + out.local_logits.square().mean()
    loss.backward()
    for group in groups:
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in group["params"])
    model.eval()
    with torch.no_grad():
        expected = model(*args).logits
    clone = build_model(arm, settings, config).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        actual = clone(*args).logits
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)


def prediction_data(n=24):
    rng = np.random.default_rng(5)
    y = np.tile(np.arange(n) % 2, (12, 1)).T.astype(float)
    return {"uids": np.asarray([str(i) for i in range(n)]), "target": y,
            "weight": np.ones_like(y), "prediction": rng.random(y.shape)}


def test_pairing_aligns_uids_but_refuses_changed_teacher_masks():
    a = prediction_data()
    b = {k: v[::-1].copy() for k, v in a.items()}
    aligned = align_pair(a, b, a["uids"].tolist())
    np.testing.assert_array_equal(aligned[0]["prediction"], aligned[1]["prediction"])
    b["weight"][0, 0] = 0
    with pytest.raises(ValueError, match="weight"):
        align_pair(a, b, a["uids"].tolist())
    b = {k: v.copy() for k, v in a.items()}
    b["uids"][0] = "missing"
    with pytest.raises(ValueError, match="UIDs"):
        align_pair(a, b, a["uids"].tolist())


def test_paired_bootstrap_identical_models_has_zero_delta_and_seed_reproducibility():
    a = prediction_data()
    groups = np.repeat(np.arange(6), 4)
    result = bootstrap_delta(a["target"], a["weight"], a["prediction"], a["prediction"],
                             groups=groups, replicates=40, seed=1)
    assert result["ci95"] == [0., 0.]
    assert result["valid_replicates"] == 40
    result2 = bootstrap_delta(a["target"], a["weight"], a["prediction"], a["prediction"],
                              groups=groups, replicates=40, seed=1)
    assert result == result2


def test_ensemble_and_expert_gates_cannot_promote_identical_models(config):
    a = prediction_data()
    config["bootstrap_replicates"] = 30
    result = compare(a, a, profiles=np.repeat(np.arange(6), 4), config=config)
    assert result["delta_vs_reference"]["candidate"] == 0
    assert not result["ensemble_meets_development_gate"]
    b = {k: v.copy() for k, v in a.items()}
    b["prediction"] = b["target"]
    expert = compare(a, b, profiles=a["uids"], config=config, expert=True)
    assert expert["delta_vs_reference"]["candidate"] > 0
    assert not expert["candidate_meets_development_gate"]
    assert not expert["ensemble_meets_development_gate"]


def test_refuses_checkpoint_ancestry_config_or_input_change():
    expected = {"version": VERSION, "arm": ARMS[1], "protocol": "same", "epochs": 12}
    require_same_run(dict(expected), expected)
    for key in expected:
        changed = {**expected, key: "different"}
        with pytest.raises(ValueError, match="contract mismatch"):
            require_same_run(changed, expected)


def test_public_initialization_refuses_competition_checkpoint_and_tampering(tmp_path):
    from rsna_knee.b57_models import DINO_URL
    from rsna_knee.b57_protocol import sha256_file
    arm = ARMS[1]
    path = tmp_path / f"{arm}.pt"
    torch.save({"experiment": "B52", "model_state": {}}, path)
    metadata = {"version": VERSION, "arm": arm, "source_url": DINO_URL,
                "competition_training_studies": [], "sha256": sha256_file(path)}
    (tmp_path / f"{arm}.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="bare encoder"):
        public_weights(tmp_path, arm)
    torch.save({"x": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        public_weights(tmp_path, arm)
    metadata["sha256"] = sha256_file(path)
    (tmp_path / f"{arm}.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="pinned public"):
        public_weights(tmp_path, arm)
    metadata["competition_training_studies"] = ["validation-study"]
    (tmp_path / f"{arm}.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="public encoder"):
        public_weights(tmp_path, arm)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, uids):
        self.study_uids = list(uids)
        self.weights = np.ones((len(uids), 12), np.float32)
        self.series_records = {u: [1] for u in uids}

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, i):
        x = torch.full((2, 3, 14, 14), .1 + .2 * (i % 2))
        return {"study_uid": self.study_uids[i], "volumes": [x], "present": torch.ones(1),
                "series_meta": torch.tensor([[1, 1, 1]]), "slice_position": torch.tensor([[0., 1.]]),
                "target": torch.full((12,), .85 if i % 2 else .05), "weight": torch.ones(12)}


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 8)
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(8, 12)

    def forward(self, volumes, present, meta, position):
        x = volumes[0].mean().reshape(1, 1)
        return SimpleNamespace(logits=self.head(self.dropout(self.encoder(x))))


def test_full_trainer_resume_matches_uninterrupted_weights_and_predictions(tmp_path, monkeypatch, config):
    from rsna_knee import b57_training as t
    c = {**config, "epochs": 3}
    splits = {"train": ["t0", "t1", "t2", "t3"], "validation": ["v0", "v1"]}
    p = {"config": c, "b42_config": {}, "splits": splits}
    contract = {"version": VERSION, "arm": ARMS[1], "epochs": 3, "protocol": "synthetic-only"}
    monkeypatch.setattr(t, "load_protocol", lambda *a, **k: p)
    monkeypatch.setattr(t, "run_contract", lambda *a, **k: contract)
    monkeypatch.setattr(t, "build_model", lambda *a, **k: SyntheticModel())
    monkeypatch.setattr(t, "make_dataset", lambda _r, _p, split: SyntheticDataset(
        ["e0", "e1"] if split == "expert" else splits[split]))
    for name in ("uninterrupted", "resumed"):
        root = tmp_path / name
        root.mkdir()
        (root / f"{ARMS[1]}_preflight.json").write_text(json.dumps({"passed": True, "contract": contract}))
    first = tmp_path / "uninterrupted"
    t.train_arm(run_root=first, arm=ARMS[1], device="cpu")
    actual_save = t.save_checkpoint

    def interrupt_after_epoch_one(*args, **kwargs):
        result = actual_save(*args, **kwargs)
        if kwargs["epoch"] == 1:
            raise RuntimeError("simulated crash after atomic checkpoint")
        return result

    monkeypatch.setattr(t, "save_checkpoint", interrupt_after_epoch_one)
    second = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated crash"):
        t.train_arm(run_root=second, arm=ARMS[1], device="cpu")
    monkeypatch.setattr(t, "save_checkpoint", actual_save)
    t.train_arm(run_root=second, arm=ARMS[1], device="cpu")
    a = torch.load(first / ARMS[1] / "final.pt", weights_only=False)
    b = torch.load(second / ARMS[1] / "final.pt", weights_only=False)
    assert a["completed_epochs"] == b["completed_epochs"] == 3
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    for split in ("validation", "expert"):
        with np.load(first / ARMS[1] / f"{split}.npz") as f, np.load(second / ARMS[1] / f"{split}.npz") as g:
            np.testing.assert_array_equal(f["prediction"], g["prediction"])
    # A completed rerun returns the same path without training or replacement.
    before = (second / ARMS[1] / "final.pt").read_bytes()
    t.train_arm(run_root=second, arm=ARMS[1], device="cpu")
    assert (second / ARMS[1] / "final.pt").read_bytes() == before


def test_prediction_loop_restores_training_mode_and_disables_autograd():
    from rsna_knee.b57_training import make_loader, predict, runtime_for
    model = SyntheticModel().train()
    seen = []
    hook = model.register_forward_hook(lambda _m, _a, out: seen.append(out.logits.requires_grad))
    runtime = runtime_for("cpu", 0)
    output = predict(model, make_loader(SyntheticDataset(["a", "b"]), runtime,
                                        {"seed": 2026, "batch_size": 2}), runtime)
    hook.remove()
    assert model.training and seen == [False, False]
    assert output["prediction"].shape == (2, 12)


def test_prepare_protocol_freezes_and_rejects_changed_inputs(tmp_path, monkeypatch, config):
    from rsna_knee import b12_variable_series, data, phase9_supervision
    from rsna_knee.b13_training import B13_SERIES_SIGNATURE
    from rsna_knee.b57_protocol import load_protocol, prepare_protocol, sha256_file
    from rsna_knee.constants import TARGETS
    # Substitute only external data/metadata loading. The production gate,
    # hash, UID, scanner, label-coverage and serialization code executes.
    root = tmp_path / "data"
    labels = tmp_path / "labels"
    gate = tmp_path / "gate"
    for directory in (root, labels, gate):
        directory.mkdir()
    uids = [f"u{i}" for i in range(8)]
    train = pd.DataFrame({"StudyInstanceUID": uids + ["e0", "e1"], "Report": ["report"] * 10})
    for target in TARGETS:
        train[target] = [np.nan] * 8 + [0., 1.]
    train.to_csv(root / "train.csv", index=False)
    (root / "train_series.csv").write_text("placeholder\n")
    (labels / "training_targets.csv").write_text("StudyInstanceUID\n" + "\n".join(uids) + "\n")
    (labels / "policy.json").write_text("{}")
    (labels / "audit.json").write_text(json.dumps({"base_cells_overridden": 0,
                                                   "gold_rows_in_training_targets": 0}))
    rows = pd.DataFrame({"StudyInstanceUID": uids,
        "scanner_profile": ["a", "a", "a", "a", "v", "v", "v", "a"],
        "parent_b48_split": ["train"] * 6 + ["validation_seen_scanners"] * 2,
        "b50_split": ["train", "train", "validation_seen_scanners", "train",
                      "validation_unseen_scanners", "validation_unseen_scanners",
                      "excluded_prior_surface", "excluded_prior_surface"]})
    rows.to_csv(gate / "b50_selection_split_by_study.csv", index=False)
    gp = {"source_train_csv_sha256": sha256_file(root / "train.csv"),
          "source_training_targets_sha256": sha256_file(labels / "training_targets.csv"),
          "unseen_scanner_profiles": ["v"]}
    (gate / "b50_selection_split.json").write_text(json.dumps(gp))
    (gate / "b50_selection_split.sha256").write_text(sha256_file(gate / "b50_selection_split.json"))
    policy = tmp_path / "series_policy.json"
    policy.write_text(json.dumps({"series_summary": {"series_signature_sha256": B13_SERIES_SIGNATURE},
                                  "policy": "all_repaired_anatomical_series_v1",
                                  "uses_gold_labels": False, "viability_passed": True,
                                  "b6_active_studies": 3120, "b6_usable_cells": 14123}))
    repo = Path(__file__).resolve().parents[2]
    target = np.tile(np.arange(8) % 2, (12, 1)).T.astype(np.float32)
    weight = np.ones_like(target)
    monkeypatch.setattr(phase9_supervision, "prepare_all_report_only_supervision",
                        lambda *_: (uids, target, weight, {"usable_cells": 34010}))
    monkeypatch.setattr(data, "load_series_csv", lambda _: pd.DataFrame())
    monkeypatch.setattr(data, "backfill_series_metadata", lambda frame, *_a, **_k: (frame, {}))
    monkeypatch.setattr(b12_variable_series, "audit_variable_series_surface",
                        lambda _s, names: ({}, {u: [{"series_uid": u + "-s"}] for u in names}))
    kwargs = dict(data_root=root, labels_root=labels, domain_split=gate, series_policy=policy,
                  b42_config=repo / "config/b42_constant_area_aspect_sparse.yaml",
                  b57_config=repo / "config/b57_clean_backbone_comparison.json",
                  out_root=tmp_path / "protocol")
    p = prepare_protocol(**kwargs)
    assert p["counts"] == {"train": 5, "validation": 2, "excluded_profile_overlap": 1}
    assert p["competition_supervised_ancestor"] is False
    before = (kwargs["out_root"] / "protocol.json").read_bytes()
    assert prepare_protocol(**kwargs) == p
    assert (kwargs["out_root"] / "protocol.json").read_bytes() == before
    (labels / "training_targets.csv").write_text("changed-teacher")
    with pytest.raises(ValueError, match="source input changed"):
        load_protocol(kwargs["out_root"])
    with pytest.raises(ValueError, match="source input changed"):
        prepare_protocol(**kwargs)


def test_bootstrap_refuses_claiming_interval_from_one_scanner():
    p = prediction_data()
    result = bootstrap_delta(p["target"], p["weight"], p["prediction"], p["prediction"],
                             groups=["same"] * len(p["uids"]), replicates=20)
    assert result["ci95"] is None
