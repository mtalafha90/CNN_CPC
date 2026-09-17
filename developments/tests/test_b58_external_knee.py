"""Exercise external-data boundaries, real pixel formats, SSL learning and transfer."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# h5py lives in the optional `b58` extra, not in `test`. Imported at module
# level it does not skip this file -- pytest interrupts collection and none of
# the repository's ~2,350 tests run, which silently removes the regression gate
# this module's own documentation relies on. `b58_external_knee/data.py`
# already gets this right, importing h5py inside `read_volume`.
h5py = pytest.importorskip("h5py", reason="h5py is in the optional b58 extra")

from rsna_knee.b58_external_knee import data, protocol, ssl
from rsna_knee.b57_protocol import digest, write_json, sha256_file


@pytest.fixture
def config():
    c = json.loads((Path(__file__).parents[2]/"config/b58_external_knee_pretraining.json").read_text())
    c.update(ssl_side=28, ssl_prototypes=32, ssl_trainable_blocks=2,
             ssl_steps=4, ssl_save_every=2, max_groups_per_source=2, max_series_per_group=2,
             cache_centres=3)
    return c


def volume(seed=1):
    return np.random.default_rng(seed).uniform(0, 100, (8, 12, 18)).astype(np.float32)


def test_mrnet_never_indexes_validation_or_test(tmp_path):
    for split in ("train", "valid", "test"):
        for plane in ("axial", "coronal", "sagittal"):
            folder = tmp_path/split/plane
            folder.mkdir(parents=True)
            np.save(folder/"0001.npy", volume())
    rows = data.index_mrnet(tmp_path)
    assert len(rows) == 3 and {r["group"] for r in rows} == {"0001"}
    assert all(Path(r["paths"][0]).parts[-3] == "train" for r in rows)
    with pytest.raises(ValueError, match="training directory"):
        data.index_mrnet(tmp_path/"valid")


def test_fastmri_reads_reconstructed_pixels_and_rejects_test_kspace(tmp_path):
    train = tmp_path/"knee_multicoil_train"
    train.mkdir()
    with h5py.File(train/"scan.h5", "w") as f:
        f.create_dataset("reconstruction_rss", data=volume())
        f.create_dataset("kspace", data=np.zeros((1, 2, 3, 4), complex))
    row = data.index_fastmri(tmp_path)[0]
    assert np.array_equal(data.read_volume(row), volume())
    with h5py.File(train/"scan.h5", "w") as f:
        f.create_dataset("kspace", data=np.zeros((1, 2, 3, 4), complex))
    with pytest.raises(ValueError, match="reconstruction_rss"):
        data.read_volume(row)


def write_dicom(path, *, patient="P1", study="1.2.3", series="1.2.3.4", position=0, value=1,
                description="SAG_3D_DESS_WE", instance=None):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID = patient, study, series
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.SeriesDescription = "MR", description
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]  # normal is x, not z
    ds.ImagePositionPatient = [position, 0, 0]
    ds.InstanceNumber = instance if instance is not None else int(position)+1
    ds.Rows, ds.Columns = 6, 8
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.full((6, 8), value, dtype=np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def test_oai_groups_visits_and_knees_and_uses_physical_sorting(tmp_path):
    for visit in range(2):
        for i in range(3):
            write_dicom(tmp_path/f"visit{visit}"/f"{2-i}.dcm", patient="P1", study=f"1.2.{visit+1}",
                        series=f"1.2.{visit+1}.4", position=i, value=10+i, instance=3-i)
    rows = data.index_oai(tmp_path)
    assert len(rows) == 2 and {r["group"] for r in rows} == {"P1"}
    x = data.read_volume(rows[0])
    assert x[:, 0, 0].tolist() == [10, 11, 12]
    with pytest.raises(ValueError, match="overlaps an RSNA"):
        data.index_oai(tmp_path, forbidden_studies=["1.2.1"])


def test_duplicate_physical_slices_and_corrupt_dicom_are_not_silently_skipped(tmp_path):
    paths = [tmp_path/f"{i}.dcm" for i in range(3)]
    for i, path in enumerate(paths):
        write_dicom(path, position=0, value=i)
    with pytest.raises(ValueError, match="duplicate DICOM slice positions"):
        data.dicom_volume(paths)
    paths[0].write_text("broken dicom")
    with pytest.raises(ValueError, match="invalid OAI DICOM"):
        data.index_oai(tmp_path)


def test_oai_excludes_thigh_and_scout():
    from types import SimpleNamespace
    assert not data.oai_knee_header(SimpleNamespace(BodyPartExamined="KNEE", SeriesDescription="SCOUT"))
    assert not data.oai_knee_header(SimpleNamespace(SeriesDescription="AX_T1_THIGH"))
    assert data.oai_knee_header(SimpleNamespace(SeriesDescription="SAG_3D_DESS_WE"))


def rows_for_sources(tmp_path, config):
    rows = []
    for si, source in enumerate(data.SOURCES):
        for group in range(3):
            for series in range(3):
                path = tmp_path/f"{source}_{group}_{series}.npy"
                np.save(path, volume(si*100+group*10+series))
                rows.append(data.record(source, f"g{group}", f"g{group}s{series}", [path], "npy"))
    return rows


def test_selection_requires_all_sources_and_caps_participants_before_series(tmp_path, config):
    rows = rows_for_sources(tmp_path, config)
    selected, counts = data.select_records(rows, config)
    assert len(selected) == 16
    assert all(c["selected_groups"] == 2 and c["selected_series"] == 4 for c in counts.values())
    assert data.select_records(list(reversed(rows)), config) == (selected, counts)
    with pytest.raises(ValueError, match="missing oai"):
        data.select_records([r for r in rows if r["source"] != "oai"], config)


def test_cache_retains_three_distinct_neighbours_and_fails_on_changed_input(tmp_path, config):
    x = np.stack([np.full((12, 20), i, np.float32) for i in range(8)])
    cached = data.cache_triplets(x, centres=3, side=28)
    assert cached.shape == (3, 3, 28, 28) and cached.dtype == np.float16
    assert np.isfinite(cached).all() and cached.min() >= 0 and cached.max() <= 1
    assert (cached[:, 0] < cached[:, 1]).all() and (cached[:, 1] < cached[:, 2]).all()
    path = tmp_path/"raw.npy"
    np.save(path, x)
    row = data.record("mrnet", "g1", "s1", [path], "npy")
    data.build_cache([row], tmp_path/"cache", config)
    np.save(path, x+1)
    with pytest.raises(ValueError, match="cache/input changed"):
        data.build_cache([row], tmp_path/"cache", config)


def test_identical_pixels_across_datasets_are_refused(tmp_path, config):
    paths = [tmp_path/f"{i}.npy" for i in range(2)]
    for p in paths:
        np.save(p, volume())
    rows = [data.record(s, "g", "s", [p], "npy") for s, p in zip(("mrnet", "fastmri"), paths)]
    with pytest.raises(ValueError, match="duplicate cached MRI content"):
        data.build_cache(rows, tmp_path/"cache", config)


def test_sampler_balances_sources_and_reproduces_after_restart(tmp_path, config):
    selected, _ = data.select_records(rows_for_sources(tmp_path, config), config)
    cached = data.build_cache(selected, tmp_path/"cache", config)
    sampler = ssl.SourceSampler(cached, config)
    sources = [row["source"] for step in range(3) for row, _ in sampler.choices(step)]
    assert {s: sources.count(s) for s in data.SOURCES} == {"rsna": 12, "mrnet": 4, "fastmri": 4, "oai": 4}
    a = sampler.batch(12)
    b = ssl.SourceSampler(cached, config).batch(12)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
    assert not torch.equal(a[0], a[1])
    assert torch.isfinite(a[0]).all()


def test_rsna_ssl_index_excludes_validation_experts_and_old_profile_overlap(tmp_path):
    p = {"data_root": str(tmp_path), "splits": {"train": ["train"], "validation": ["val"],
         "excluded_profile_overlap": ["excluded"]}, "expert_uids": ["expert"]}
    index = {u: [{"series_uid": "s"}] for u in ("train", "val", "excluded", "expert")}
    for u in index:
        for i in range(3):
            write_dicom(tmp_path/"train_series"/u/"s"/f"{i}.dcm", position=i)
    write_json(tmp_path/"series_index.json", index)
    rows = data.index_rsna(p, tmp_path)
    assert {r["group"] for r in rows} == {"train"}
    p["splits"]["train"].append("val")
    with pytest.raises(ValueError, match="reached the SSL index"):
        data.index_rsna(p, tmp_path)


def test_portable_rsna_identity_ignores_mount_paths_but_not_labels(tmp_path):
    p = {"inputs": {"a": {"path": "/machine1/data", "sha256": "123"}},
         "split_uid_hashes": {"train": "x"}, "expert_uids": [], "series_index_sha256": "index",
         "scanner_profiles": {"u": "s"}, "config": {"epochs": 12}, "b42_config": {"area": 448**2}}
    np.savez(tmp_path/"labels.npz", uids=np.array(["u"]), target=np.ones((1, 12)), weight=np.ones((1, 12)))
    before = protocol.rsna_identity(tmp_path, p)
    p["inputs"]["a"]["path"] = "/machine2/mount"
    assert before == protocol.rsna_identity(tmp_path, p)
    np.savez(tmp_path/"labels.npz", uids=np.array(["u"]), target=np.zeros((1, 12)), weight=np.ones((1, 12)))
    assert before != protocol.rsna_identity(tmp_path, p)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.patch = nn.Conv2d(3, 8, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.GELU()) for _ in range(3)])
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.patch(x).mean((2, 3))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def test_real_ssl_gradients_teacher_and_frozen_blocks_and_resume(config):
    torch.manual_seed(7)
    model = ssl.Adaptation(TinyEncoder(), config).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    inputs = [torch.rand(8, 3, 28, 28) for _ in range(2)]
    old = deepcopy(model.state_dict())
    loss, center, stats = model.loss(*inputs)
    loss.backward()
    assert all(v > 0 for v in ssl.check_gradients(model).values())
    assert model.student.patch.weight.grad is None and model.student.blocks[0][0].weight.grad is None
    optimizer.step()
    model.update_teacher(center, 1)
    assert torch.equal(old["student.patch.weight"], model.student.patch.weight)
    assert not torch.equal(old["student.blocks.2.0.weight"], model.student.blocks[2][0].weight)
    assert not torch.equal(old["teacher.blocks.2.0.weight"], model.teacher.blocks[2][0].weight)
    assert torch.isfinite(model.center).all() and torch.count_nonzero(model.center)
    resumed = ssl.Adaptation(TinyEncoder(), config).train()
    resumed.load_state_dict(model.state_dict(), strict=True)
    resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.001)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for m, opt in ((model, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        loss, center, _ = m.loss(*inputs)
        loss.backward()
        opt.step()
        m.update_teacher(center, 2)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in model.state_dict().items())


def test_real_dinov2_adaptation_and_strict_transfer(config):
    from rsna_knee.b57_models import dino_backbone, DinoSliceTransformer
    model = ssl.Adaptation(dino_backbone(pretrained=False), config).train()
    loss, center, _ = model.loss(torch.rand(2, 3, 28, 28), torch.rand(2, 3, 28, 28))
    loss.backward()
    assert ssl.check_gradients(model)["encoder"] > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001)
    optimizer.step()
    model.update_teacher(center, 1)
    downstream = DinoSliceTransformer(dino_backbone(pretrained=False), dim=384)
    downstream.encoder.load_state_dict(model.teacher.state_dict(), strict=True)
    assert all(torch.equal(v, downstream.encoder.state_dict()[k]) for k, v in model.teacher.state_dict().items())


def test_config_requires_all_sources_and_patch_compatible_resolution(config):
    protocol.validate_config(config)
    config["sources"] = ["rsna", "mrnet"]
    with pytest.raises(ValueError, match="identity/source"):
        protocol.validate_config(config)


def test_protocol_refuses_mutated_cache_and_source(tmp_path, config, monkeypatch):
    # Build a minimal frozen artifact boundary to exercise the actual loader.
    rsna = {"dummy": True}
    root = tmp_path
    config_path = root/"config.json"
    write_json(config_path, config)
    write_json(root/"selection.json", [])
    cache = root/"cache.npy"
    np.save(cache, np.ones((3, 3, 28, 28), np.float16))
    write_json(root/"cache_manifest.json", [{"cache": str(cache), "cache_sha256": sha256_file(cache)}])
    write_json(root/"rsna_protocol/protocol.json", rsna)
    p = {"version": protocol.VERSION, "config": config, "rsna_identity": "identity",
         "request": {"implementation": "source", "config_path": str(config_path), "config_sha256": sha256_file(config_path)},
         "selection_sha256": sha256_file(root/"selection.json"),
         "cache_manifest_sha256": sha256_file(root/"cache_manifest.json"),
         "rsna_protocol_sha256": sha256_file(root/"rsna_protocol/protocol.json")}
    write_json(root/"protocol.json", p)
    (root/"protocol.sha256").write_text(sha256_file(root/"protocol.json")+"\n")
    monkeypatch.setattr(protocol, "implementation_digest", lambda: "source")
    monkeypatch.setattr(protocol, "load_rsna", lambda _: rsna)
    monkeypatch.setattr(protocol, "rsna_identity", lambda *_: "identity")
    assert protocol.load(root, verify_cache=True)["version"] == protocol.VERSION
    np.save(cache, np.zeros((3, 3, 28, 28), np.float16))
    with pytest.raises(ValueError, match="cached pixels changed"):
        protocol.load(root, verify_cache=True)
    monkeypatch.setattr(protocol, "implementation_digest", lambda: "new source")
    with pytest.raises(ValueError, match="source changed"):
        protocol.load(root)


def test_ssl_trainer_restart_preserves_weights_and_exact_source_exposure(tmp_path, config, monkeypatch):
    selected, _ = data.select_records(rows_for_sources(tmp_path, config), config)
    cached = data.build_cache(selected, tmp_path/"cache", config)
    p = {"config": config, "rsna_identity": "same-rsna"}
    monkeypatch.setattr(ssl, "load", lambda *a, **k: p)

    def build(*_):
        torch.manual_seed(config["seed"])
        return ssl.Adaptation(TinyEncoder(), config)

    monkeypatch.setattr(ssl, "build", build)
    roots = [tmp_path/name for name in ("continuous", "interrupted")]
    for root in roots:
        write_json(root/"protocol.json", p)
        write_json(root/"cache_manifest.json", cached)
        ssl.preflight(root, device="cpu")
    ssl.train(roots[0], device="cpu")
    original_save = ssl.atomic_torch_save

    def interrupt(path, payload):
        original_save(path, payload)
        if Path(path).name == "recovery_latest.pt" and payload["step"] == 2:
            raise RuntimeError("simulated SSL interruption")

    monkeypatch.setattr(ssl, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated SSL"):
        ssl.train(roots[1], device="cpu")
    monkeypatch.setattr(ssl, "atomic_torch_save", original_save)
    ssl.train(roots[1], device="cpu")
    a, ma = ssl.load_encoder(roots[0], p)
    b, mb = ssl.load_encoder(roots[1], p)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert ma["source_samples"] == mb["source_samples"] == {
        "rsna": 16, "mrnet": 6, "fastmri": 5, "oai": 5}
    path = roots[1]/"pretrain/encoder.pt"
    before = path.read_bytes()
    ssl.train(roots[1], device="cpu")
    assert path.read_bytes() == before
    mb["source_samples"]["oai"] -= 1
    write_json(roots[1]/"pretrain/complete.json", mb)
    with pytest.raises(ValueError, match="exposure mismatch"):
        ssl.load_encoder(roots[1], p)


@pytest.mark.parametrize("revision", ["v1", "full_data_v2"])
def test_full_supervised_trainer_recovers_and_exports_comparable_predictions(tmp_path, monkeypatch, revision):
    from types import SimpleNamespace
    if revision == "v1":
        from rsna_knee.b58_external_knee import training as t, evaluation as e, VERSION
    else:
        from rsna_knee.b58_full_data import finetune as t, evaluation as e, VERSION
    from rsna_knee.b57_models import ARMS
    from rsna_knee.b57_protocol import VERSION as B57_VERSION
    from rsna_knee.b57_training import save_predictions as export_b57

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, uids):
            self.study_uids = uids
            self.weights = np.ones((len(uids), 12), np.float32)
            self.series_records = {u: [1] for u in uids}

        def __len__(self):
            return len(self.study_uids)

        def __getitem__(self, i):
            return {"study_uid": self.study_uids[i],
                    "volumes": [torch.full((2, 3, 14, 14), .1+.2*(i % 2))],
                    "present": torch.ones(1), "series_meta": torch.tensor([[1, 1, 1]]),
                    "slice_position": torch.tensor([[0., 1.]]),
                    "target": torch.full((12,), .85 if i % 2 else .05), "weight": torch.ones(12)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(1, 8)
            self.dropout = nn.Dropout(.1)
            self.head = nn.Linear(8, 12)

        def forward(self, volumes, present, meta, position):
            return SimpleNamespace(logits=self.head(self.dropout(self.encoder(volumes[0].mean().reshape(1, 1)))))

    c = json.loads((Path(__file__).parents[2]/"config/b57_clean_backbone_comparison.json").read_text())
    c["epochs"] = 3
    splits = {"train": ["t0", "t1", "t2", "t3"], "validation": ["v0", "v1"]}
    rsna = {"config": c, "b42_config": {}, "splits": splits, "expert_uids": ["e0", "e1"],
            "scanner_profiles": {"v0": "scanner0", "v1": "scanner1"}}
    p = {"rsna_identity": "same-rsna", "config": {"seed": 2026, "bootstrap_replicates": 10,
                                                   "candidate_min_delta": .01}}
    monkeypatch.setattr(t, "load", lambda *a, **k: p)
    monkeypatch.setattr(t, "load_protocol", lambda *a, **k: rsna)
    monkeypatch.setattr(t, "make_dataset", lambda _r, _p, split:
                        Dataset(rsna["expert_uids"] if split == "expert" else splits[split]))

    def construct(*a, **k):
        torch.manual_seed(c["seed"])
        return Model()

    def contract(root, *_):
        return {"version": VERSION, "rsna_identity": p["rsna_identity"], "epochs": 3,
                "protocol_sha256": sha256_file(Path(root)/"protocol.json"),
                "finetune_config_sha256": digest(c), "gold_studies_in_gradient": 0,
                "training_uids_sha256": digest(splits["train"]),
                "validation_uids_sha256": digest(splits["validation"])}

    monkeypatch.setattr(t, "construct", construct)
    monkeypatch.setattr(t, "fine_contract", contract)
    roots = [tmp_path/name for name in ("continuous", "interrupted")]
    for root in roots:
        write_json(root/"protocol.json", p)
        t.preflight(root, device="cpu", adapted=True)
    t.train(roots[0], device="cpu")
    original_save = t.save_checkpoint

    def interrupt(*a, **k):
        original_save(*a, **k)
        if k["epoch"] == 1:
            raise RuntimeError("simulated finetune interruption")

    monkeypatch.setattr(t, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="simulated finetune"):
        t.train(roots[1], device="cpu")
    monkeypatch.setattr(t, "save_checkpoint", original_save)
    t.train(roots[1], device="cpu")
    states = [torch.load(r/"finetune/final.pt", weights_only=False)["model_state"] for r in roots]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    for split in ("validation", "expert"):
        a, _ = e.candidate_predictions(roots[0], split, p)
        b, _ = e.candidate_predictions(roots[1], split, p)
        np.testing.assert_array_equal(a["prediction"], b["prediction"])
    before = (roots[1]/"finetune/final.pt").read_bytes()
    t.train(roots[1], device="cpu")
    assert (roots[1]/"finetune/final.pt").read_bytes() == before

    # Exercise actual B57 and B58 export readers, pairing, frozen-label checks
    # and comparison JSON using a portable completed reference fixture.
    reference, candidate = tmp_path/"b57_export", roots[1]
    write_json(reference/"protocol/protocol.json", {"export": True})
    directory = reference/ARMS[1]
    directory.mkdir()
    (directory/"final.pt").write_bytes(b"synthetic checkpoint; reader verifies bytes without deserializing")
    reference_contract = contract(candidate) | {"version": B57_VERSION, "arm": ARMS[1],
        "protocol_sha256": sha256_file(reference/"protocol/protocol.json"),
        "competition_supervised_ancestor": False, "config_sha256": digest(c)}
    checkpoint_sha = sha256_file(directory/"final.pt")
    write_json(directory/"complete.json", {"contract": reference_contract, "completed_epochs": 3,
                                           "checkpoint_sha256": checkpoint_sha})
    values = {}
    for split in ("validation", "expert"):
        values[split], _ = e.candidate_predictions(candidate, split, p)
        export_b57(directory/f"{split}.npz", values[split], contract=reference_contract,
                   split=split, checkpoint_sha=checkpoint_sha)
    (candidate/"rsna_protocol").mkdir()
    np.savez(candidate/"rsna_protocol/labels.npz", uids=values["validation"]["uids"],
             target=values["validation"]["target"], weight=values["validation"]["weight"],
             expert_uids=values["expert"]["uids"], expert_target=values["expert"]["target"])
    monkeypatch.setattr(e, "load", lambda *a, **k: p)
    monkeypatch.setattr(e, "verify_reference_boundary", lambda *a: (rsna, rsna))
    result = e.evaluate(candidate, reference)
    assert result["validation"]["delta"] == result["expert"]["delta"] == 0
    assert result["automatic_promotion_or_submission"] is False
    # Even two internally consistent exports cannot override frozen labels.
    values["validation"]["target"][0, 0] = .95
    np.savez(candidate/"rsna_protocol/labels.npz", uids=values["validation"]["uids"],
             target=values["validation"]["target"], weight=values["validation"]["weight"],
             expert_uids=values["expert"]["uids"], expert_target=values["expert"]["target"])
    with pytest.raises(ValueError, match="frozen label/mask"):
        e.evaluate(candidate, reference)


def test_cli_run_stops_after_failed_stage(tmp_path, monkeypatch):
    import subprocess
    from rsna_knee.b58_external_knee import __main__ as cli
    write_json(tmp_path/"protocol.json", {})
    seen = []

    def fail(cmd, check):
        assert check
        seen.append(cmd[3])
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cli.main(["run", "--run-root", str(tmp_path), "--device", "cpu"])
    assert seen == ["preflight"]
