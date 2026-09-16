"""Full data/label coverage, real gradients, provenance and interrupted runs."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee.b57_protocol import digest, sha256_file, write_json
from rsna_knee.b58_external_knee.data import record, index_mrnet, SOURCES
from rsna_knee.b58_full_data import VERSION
from rsna_knee.b58_full_data import data, labels, models, protocol, training


def csv_file(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


@pytest.fixture
def config():
    c = json.loads((Path(__file__).parents[2] / "config/b58_full_data.json").read_text())
    c.update(ssl_side=28, supervised_side=28, ssl_batch_size=2, ssl_prototypes=8,
             ssl_trainable_blocks=2, ssl_epochs=2, supervised_epochs=2, save_every=1, slices_per_series=3)
    return c


def mri_rows(tmp_path, *, per_source=2):
    result = []
    for s, source in enumerate(SOURCES):
        for i in range(per_source):
            path = tmp_path / f"{source}_{i}.npy"
            np.save(path, np.random.default_rng(s*10+i).uniform(0, 100, (6, 8, 12)).astype(np.float32))
            result.append(record(source, f"g{i}", f"s{i}", [path], "npy"))
    return result


@pytest.mark.parametrize("promoted_header", [False, True])
def test_mrnet_full_labels_and_redivis_first_observation(tmp_path, promoted_header):
    for plane in ("axial", "sagittal", "coronal"):
        folder = tmp_path / "train" / plane
        folder.mkdir(parents=True)
        for uid in range(3):
            np.save(folder / f"{uid:04d}.npy", np.zeros((3, 3, 3)))
    for name in ("abnormal", "acl", "meniscus"):
        first = "_0000,_1" if promoted_header else "0000,1"
        (tmp_path / f"train-{name}.csv").write_text(first + "\n0001,0\n0002,1\n")
    rows = index_mrnet(tmp_path)
    tasks, table, _, report = labels.mrnet_labels(tmp_path, rows, expected_exams=3)
    assert len(tasks) == 3 and len(table) == 3 and table["0000"]["values"] == [1, 1, 1]
    assert len(report["recovered_rows"]) == (3 if promoted_header else 0)
    (tmp_path / "train-acl.csv").write_text("0001,0\n0002,1\n")
    with pytest.raises(ValueError, match="all 3 training exams"):
        labels.mrnet_labels(tmp_path, rows, expected_exams=3)


def test_fastmri_unknown_is_masked_and_reviewed_normal_is_negative(tmp_path):
    annotation = csv_file(tmp_path / "knee.csv", ["file", "slice", "study_level", "label"],
                          [["positive", "1", "No", "Meniscus Tear"], ["heldout", "1", "No", "Joint Effusion "]])
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("positive\nnormal\nheldout\n")
    rows = [{"group": n} for n in ("positive", "normal", "unknown")]
    tasks, y, _, audit = labels.fastmri_labels(annotation, reviewed, rows)
    assert [t["name"] for t in tasks] == ["Joint Effusion", "Meniscus Tear"]
    assert y["positive"] == {"values": [0., 1.], "weights": [0., 1.]}
    assert y["normal"] == {"values": [0., 0.], "weights": [1., 1.]}
    assert y["unknown"]["weights"] == [0., 0.]
    assert "heldout" not in y and audit["unreviewed"] == 1
    reviewed.write_text("normal\n")
    with pytest.raises(ValueError, match="review identity"):
        labels.fastmri_labels(annotation, reviewed, rows)


def oai_fixture(tmp_path):
    task_path = tmp_path / "tasks.json"
    tasks = [{"name": "native_grade", "kind": "categorical", "classes": ["0", "1", "2"],
              "missing_values": ["-1"], "provenance": "fixture dictionary"},
             {"name": "native_score", "kind": "regression", "minimum": 0, "maximum": 10,
              "provenance": "fixture dictionary"}]
    write_json(task_path, {"tasks": tasks})
    mapping = csv_file(tmp_path / "series.csv", ["patient_id", "visit", "side", "series_uid", "partition"],
        [["p1", "V00", "R", "s1", "train"], ["p1", "V01", "L", "s2", "train"],
         ["p2", "V00", "R", "s3", "validation"]])
    annotation = csv_file(tmp_path / "labels.csv", ["patient_id", "visit", "side", "native_grade", "native_score"],
        [["p1", "V00", "R", "2", "5"], ["p1", "V01", "L", "-1", ""], ["p2", "V00", "R", "1", "8"]])
    rows = [{"group": "p1" if i < 3 else "p2", "series": f"s{i}", "kind": "npy"} for i in (1, 2, 3)]
    return mapping, annotation, task_path, rows


def test_oai_native_labels_join_exact_visit_side_and_exclude_heldout(tmp_path):
    args = oai_fixture(tmp_path)
    tasks, y, kept, _, audit = labels.oai_labels(*args)
    assert len(kept) == 2 and audit["held_out_series"] == 1
    assert y[json.dumps(["p1", "V00", "R"], separators=(",", ":"))] == {"values": [2, .5], "weights": [1., 1.]}
    assert y[json.dumps(["p1", "V01", "L"], separators=(",", ":"))]["weights"] == [0., 0.]
    assert audit["observed_training_labels_per_task"] == [1, 1]


@pytest.mark.parametrize("problem", ["leakage", "missing_join", "wrong_participant", "unknown_label", "duplicate_label", "undeclared_column"])
def test_oai_rejects_ambiguous_labels_or_boundaries(tmp_path, problem):
    mapping, annotation, schema, rows = oai_fixture(tmp_path)
    if problem == "leakage":
        mapping.write_text(mapping.read_text().replace("s2,train", "s2,test"))
    elif problem == "missing_join":
        rows.append({"group": "p3", "series": "s4", "kind": "npy"})
    elif problem == "wrong_participant":
        rows[0]["group"] = "wrong"
    elif problem == "unknown_label":
        annotation.write_text(annotation.read_text().replace("R,2,5", "R,7,5"))
    elif problem == "duplicate_label":
        with annotation.open("a") as handle:
            handle.write("p1,V00,R,2,5\n")
    else:
        annotation.write_text(annotation.read_text().replace("native_score", "native_score,extra").replace(",5\n", ",5,0\n").replace(",,\n", ",,,0\n"))
    with pytest.raises(ValueError):
        labels.oai_labels(mapping, annotation, schema, rows)


def test_full_inventory_has_no_old_caps_and_epoch_order_visits_every_item():
    rows = [{"source": s, "group": str(i), "series": f"{i}/{j}"} for s in SOURCES for i in range(1030) for j in range(3)]
    selected, counts = data.complete_inventory(rows)
    assert len(selected) == len(rows)
    assert all(c == {"series": 3090, "groups": 1030} for c in counts.values())
    for epoch in (0, 1, 4):
        order = data.epoch_order(selected, 2026, epoch)
        assert sorted(order) == list(range(len(rows)))
        assert order == data.epoch_order(selected, 2026, epoch)
    assert data.epoch_order(selected, 2026, 0) != data.epoch_order(selected, 2026, 1)


def test_streaming_freeze_resume_duplicate_and_mutation(tmp_path):
    rows = mri_rows(tmp_path)
    frozen = data.freeze_pixels(rows, tmp_path / "audit")
    assert frozen == data.freeze_pixels(rows, tmp_path / "audit")
    x = data.triplets(frozen[0], 3, 28)
    assert x.shape == (3, 3, 28, 28) and torch.isfinite(x).all()
    np.save(rows[0]["paths"][0], np.ones((6, 8, 12)))
    with pytest.raises(ValueError, match="bytes changed"):
        data.triplets(frozen[0], 3, 28)
    with pytest.raises(ValueError, match="changed during preparation"):
        data.freeze_pixels(rows, tmp_path / "audit")
    np.save(rows[0]["paths"][0], np.load(rows[1]["paths"][0]))
    with pytest.raises(ValueError, match="duplicate MRI"):
        data.freeze_pixels(rows, tmp_path / "another_audit")


def test_native_mask_has_zero_gradient_and_regression_categorical_work():
    tasks = [labels.binary_task("a"), labels.binary_task("unknown"),
             {"name": "grade", "kind": "categorical", "classes": ["0", "1", "2"]},
             {"name": "score", "kind": "regression"}]
    logits = [torch.tensor([.3], requires_grad=True), torch.tensor([-.4], requires_grad=True),
              torch.tensor([.1, .4, -.1], requires_grad=True), torch.tensor([.1], requires_grad=True)]
    loss = models.native_loss(logits, tasks, [1., 1., 2, .7], [1., 0., 1., 1.])
    loss.backward()
    assert logits[1].grad is None
    assert all(torch.count_nonzero(logits[i].grad) > 0 for i in (0, 2, 3))
    with pytest.raises(ValueError, match="unlabelled"):
        models.native_loss(logits, tasks, [0.] * 4, [0.] * 4)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.conv = nn.Conv2d(3, 8, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.GELU()) for _ in range(2)])
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.conv(x).mean((2, 3))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def small_run(root, config, monkeypatch):
    root.mkdir()
    records = data.freeze_pixels(mri_rows(root), root / "audit")
    tasks = {s: [labels.binary_task("one"), labels.binary_task("two")] for s in SOURCES}
    tables = {s: {"g0": {"values": [0., 1.], "weights": [1., 1.]},
                  "g1": {"values": [1., 0.], "weights": [1., 1.]}} for s in SOURCES}
    cases = data.build_cases(records, tables, tasks)
    manifest = {"records": records, "tasks": tasks, "cases": cases}
    write_json(root / "manifest.json", manifest)
    p = {"version": VERSION, "config": config, "rsna_identity": "fixture-rsna", "ssl_steps": len(records)*config["ssl_epochs"],
         "supervised_steps": len(cases)*config["supervised_epochs"]}
    write_json(root / "protocol.json", p)
    monkeypatch.setattr(training, "load", lambda root: json.loads((Path(root) / "protocol.json").read_text()))
    monkeypatch.setattr(training, "dino_backbone", TinyEncoder)
    torch.manual_seed(42)
    state = deepcopy(TinyEncoder().state_dict())
    monkeypatch.setattr(training, "public_weights", lambda *_: (deepcopy(state), {}))
    return p, manifest


@pytest.mark.parametrize("stage", ["pretrain", "supervised"])
def test_full_train_resume_exact_weights_and_all_sources(tmp_path, config, monkeypatch, stage):
    root = tmp_path / "run"
    p, manifest = small_run(root, config, monkeypatch)
    if stage == "supervised":
        training.preflight(root, "pretrain", device="cpu")
        training.train(root, "pretrain", device="cpu")
    check = training.preflight(root, stage, device="cpu")
    assert check["optimizer_steps"] == 0 and set(check["sources"]) == set(SOURCES)
    training.train(root, stage, device="cpu")
    continuous = torch.load(root / stage / "encoder.pt", weights_only=True)
    complete = json.loads((root / stage / "complete.json").read_text())
    assert complete["source_exposure"] == {s: 4 for s in SOURCES}
    assert complete["steps"] == 16 and complete["epochs"] == 2
    training.load_encoder(root, p, stage)
    # Restart from scratch then interrupt immediately AFTER an atomic save.
    import shutil
    shutil.rmtree(root / stage)
    original_save = training.atomic_torch_save
    def interrupted_save(path, value):
        original_save(path, value)
        if Path(path).name == "recovery_latest.pt" and value["step"] == 5:
            raise RuntimeError("power loss")
    monkeypatch.setattr(training, "atomic_torch_save", interrupted_save)
    with pytest.raises(RuntimeError, match="power loss"):
        training.train(root, stage, device="cpu")
    monkeypatch.setattr(training, "atomic_torch_save", original_save)
    torch.rand(123)  # Process age must not affect the resumed schedule or views.
    training.train(root, stage, device="cpu")
    resumed = torch.load(root / stage / "encoder.pt", weights_only=True)
    assert all(torch.equal(continuous[k], resumed[k]) for k in continuous)
    assert json.loads((root / stage / "complete.json").read_text())["source_exposure"] == complete["source_exposure"]
    # Completed source accounting is mandatory, not a decorative log.
    meta = json.loads((root / stage / "complete.json").read_text())
    meta["source_exposure"]["mrnet"] -= 1
    write_json(root / stage / "complete.json", meta)
    with pytest.raises(ValueError, match="source exposure"):
        training.load_encoder(root, p, stage)


def test_real_dinov2_all_native_heads_backward_and_strict_encoder_transfer(config, monkeypatch):
    from rsna_knee.b57_models import dino_backbone, DinoSliceTransformer
    from rsna_knee.b58_full_data import finetune
    encoder = dino_backbone()
    tasks = {s: [labels.binary_task("finding")] for s in SOURCES}
    model = models.NativeHeads(encoder, tasks, chunk_size=2).train()
    for s in SOURCES:
        model.zero_grad(set_to_none=True)
        logits = model(s, [torch.rand(2, 3, 28, 28)])
        models.native_loss(logits, tasks[s], [1.], [1.]).backward()
        assert model.encoder.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
        assert model.heads[s][0][1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    before = model.encoder.blocks[0].attn.qkv.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.encoder.blocks[0].attn.qkv.weight)
    monkeypatch.setattr(finetune, "load_encoder", lambda *_: (model.encoder.state_dict(), {}))
    monkeypatch.setattr(finetune, "build_model", lambda *_: DinoSliceTransformer(dino_backbone()))
    downstream = finetune.construct("unused", {}, {"config": {"seed": 2026}, "b42_config": {}}, adapted=True)
    assert all(torch.equal(model.encoder.state_dict()[k], downstream.encoder.state_dict()[k]) for k in model.encoder.state_dict())


def test_full_config_refuses_caps_and_only_accepts_full_encoder():
    c = json.loads((Path(__file__).parents[2] / "config/b58_full_data.json").read_text())
    protocol.validate_config(c)
    with pytest.raises(ValueError, match="caps"):
        protocol.validate_config(c | {"max_groups_per_source": 1024})
    with pytest.raises(ValueError, match="all twelve"):
        protocol.validate_config(c | {"ssl_trainable_blocks": 4})


def test_preparation_joins_every_source_freezes_labels_and_refuses_tampering(tmp_path, monkeypatch):
    rows = mri_rows(tmp_path, per_source=3)
    pools = {s: [r for r in rows if r["source"] == s] for s in SOURCES}
    for row, plane in zip(pools["mrnet"], ("axial", "sagittal", "coronal")):
        row.update(group="0000", series="0000/" + plane)
    for name in ("abnormal", "acl", "meniscus"):
        (tmp_path / f"train-{name}.csv").write_text("0000,1\n")
    annotations = csv_file(tmp_path / "knee.csv", ["file", "slice", "study_level", "label"], [["g0", "2", "No", "Meniscus Tear"]])
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("g0\ng1\n")
    mapping, oai_csv, schema, oai_rows = oai_fixture(tmp_path)
    for row, identity in zip(pools["oai"], oai_rows):
        row.update(identity)
    monkeypatch.setattr(protocol, "index_rsna", lambda *_: pools["rsna"])
    monkeypatch.setattr(protocol, "index_mrnet", lambda *_: pools["mrnet"])
    monkeypatch.setattr(protocol, "index_fastmri", lambda *_: pools["fastmri"])
    monkeypatch.setattr(protocol, "index_oai", lambda *_, **kw: pools["oai"])
    rsna = {"splits": {"train": ["g0", "g1", "g2"], "validation": ["held"]}, "counts": {"train": 3, "validation": 1}}
    def prepare_rsna(**kw):
        root = Path(kw["out_root"])
        write_json(root / "protocol.json", rsna)
        write_json(root / "series_index.json", {r["group"]: [{"series_uid": r["series"]}] for r in pools["rsna"]})
        np.savez(root / "labels.npz", uids=np.asarray(["g0", "g1", "g2", "held"]), target=np.full((4, 12), .85), weight=np.ones((4, 12)))
        return rsna
    monkeypatch.setattr(protocol, "prepare_protocol", prepare_rsna)
    monkeypatch.setattr(protocol, "load_protocol", lambda *_: rsna)
    monkeypatch.setattr(protocol, "rsna_identity", lambda *_: "frozen-rsna")
    monkeypatch.setattr(protocol, "prepare_public_weights", lambda *_: None)
    c = json.loads((Path(__file__).parents[2] / "config/b58_full_data.json").read_text())
    c["mrnet_expected_exams"] = 1
    config_path = tmp_path / "config.json"
    write_json(config_path, c)
    root = tmp_path / "run"
    kwargs = dict(run_root=root, data_root=tmp_path, labels_root=tmp_path, domain_split=tmp_path,
                  series_policy=tmp_path, mrnet_root=tmp_path, fastmri_root=tmp_path, oai_root=tmp_path,
                  fastmri_annotations=annotations, fastmri_reviewed=reviewed, oai_series=mapping,
                  oai_labels_csv=oai_csv, oai_tasks=schema, config_path=config_path)
    p = protocol.prepare(**kwargs)
    assert p == protocol.load(root) == protocol.prepare(**kwargs)
    manifest = json.loads((root / "manifest.json").read_text())
    assert p["ssl_steps"] == 11  # OAI held-out series removed, no training cap.
    assert set(p["label_coverage"]) == set(SOURCES)
    assert sum(r["source"] == "mrnet" for r in manifest["records"]) == 3
    assert all(r.get("group") != "p2" for r in manifest["records"])
    assert all(r["case_id"] != "held" for r in manifest["cases"])
    schema.write_text(schema.read_text() + "\n")
    with pytest.raises(ValueError, match="label input changed"):
        protocol.load(root)
    # Restoring a label does not authorize an edited pixel or supervision manifest.
    schema.write_text(schema.read_text()[:-1])
    manifest["cases"][0]["weights"][0] = 0
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.load(root)


def test_full_cli_stops_on_failure_and_uses_own_output_root(tmp_path, monkeypatch):
    from rsna_knee.b58_full_data import __main__ as cli, DEFAULT_ROOT
    assert cli.parser().parse_args(["run"]).run_root == DEFAULT_ROOT
    (tmp_path / "protocol.json").write_text("{}")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[3] == "pretrain":
            raise RuntimeError("failed")
    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="failed"):
        cli.main(["run", "--run-root", str(tmp_path)])
    assert [c[3] for c in calls] == ["preflight", "pretrain"]
