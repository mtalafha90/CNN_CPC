"""Reading the sparse gate out of finished runs.

The tool exists to answer one question from files already on disk: how much of
the scored prediction the local branch was actually allowed to contribute. It
must therefore find the gate wherever a completed run happened to record it, and
report what it finds without editorialising.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tools.gate_audit import collect, describe, gates_from_audit, gates_from_checkpoint


def _history(values, name="history.json"):
    return [
        {
            "epoch": 1,
            "gate": {
                "gate_raw": values,
                "gate_effective": [math.tanh(v) for v in values],
                "gate_effective_abs_mean": sum(abs(math.tanh(v)) for v in values) / len(values),
                "gate_effective_abs_max": max(abs(math.tanh(v)) for v in values),
            },
        }
    ]


def test_the_gate_is_found_where_training_recorded_it(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(_history([0.0, 0.5, -0.25])))

    rows = gates_from_audit(path)
    assert len(rows) == 1
    assert rows[0]["gate_effective"] == pytest.approx([0.0, math.tanh(0.5), math.tanh(-0.25)])
    assert rows[0]["where"].endswith("gate")


def test_a_gate_nested_anywhere_is_still_found(tmp_path):
    """Audits differ between experiments; the reader must not assume a shape."""
    path = tmp_path / "training_audit.json"
    path.write_text(
        json.dumps(
            {
                "model_state": {"head": {"gate_effective": [0.1, 0.2], "gate_raw": [0.1, 0.2]}},
                "epochs": [{"deeply": {"nested": {"gate_effective": [0.3]}}}],
            }
        )
    )
    rows = gates_from_audit(path)
    assert len(rows) == 2
    assert {len(row["gate_effective"]) for row in rows} == {1, 2}


def test_a_file_with_no_gate_yields_nothing_rather_than_failing(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"epoch": 1, "loss_total": 0.5}))
    assert gates_from_audit(path) == []


def test_unreadable_json_is_reported_not_raised(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json")
    rows = gates_from_audit(path)
    assert len(rows) == 1 and "error" in rows[0]


def test_the_gate_is_read_from_a_checkpoint_and_converted(tmp_path):
    """The checkpoint stores the raw parameter; tanh is what the model applies."""
    path = tmp_path / "model.pt"
    torch.save({"model": {"head.gate": torch.tensor([0.0, 1.0, -2.0])}}, path)

    rows = gates_from_checkpoint(path)
    assert len(rows) == 1
    assert rows[0]["gate_raw"] == pytest.approx([0.0, 1.0, -2.0])
    assert rows[0]["gate_effective"] == pytest.approx(
        [0.0, math.tanh(1.0), math.tanh(-2.0)]
    )
    assert rows[0]["abs_max"] == pytest.approx(math.tanh(2.0))


def test_the_second_gate_b48_and_b49_add_is_also_read(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": {
                "head.gate": torch.zeros(12),
                "head.context_gate": torch.full((12,), 0.75),
            }
        },
        path,
    )
    rows = gates_from_checkpoint(path)
    assert {row["where"] for row in rows} == {"head.gate", "head.context_gate"}


def test_a_bare_state_dict_is_handled(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"head.gate": torch.tensor([0.5])}, path)
    rows = gates_from_checkpoint(path)
    assert rows[0]["gate_effective"] == pytest.approx([math.tanh(0.5)])


def test_a_checkpoint_without_a_gate_says_so(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": {"encoder.weight": torch.zeros(3)}}, path)
    rows = gates_from_checkpoint(path)
    assert "no gate parameter found" in rows[0]["error"]


def test_a_whole_run_directory_is_swept(tmp_path):
    (tmp_path / "fold_0").mkdir()
    (tmp_path / "fold_0" / "history.json").write_text(json.dumps(_history([0.1, 0.2])))
    torch.save({"model": {"head.gate": torch.zeros(2)}}, tmp_path / "fold_0" / "model.pt")

    rows = collect(tmp_path)
    assert len(rows) == 2
    assert any(row["source"].endswith("history.json") for row in rows)
    assert any(row["source"].endswith("model.pt") for row in rows)


def test_a_single_file_can_be_pointed_at_directly(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(_history([0.4])))
    assert len(collect(path)) == 1


def test_the_report_names_targets_when_the_count_matches():
    rows = [
        {
            "source": "run/history.json",
            "where": "gate",
            "gate_effective": [0.01, -0.02],
            "gate_raw": [0.01, -0.02],
            "abs_mean": 0.015,
            "abs_max": 0.02,
        }
    ]
    named = describe(rows, ["ACL", "MCL"])
    assert "ACL" in named and "MCL" in named
    unnamed = describe(rows, ["ACL", "MCL", "Effusion"])  # wrong length
    assert "target 0" in unnamed


def test_the_report_states_no_verdict():
    """A threshold in the tool would get read back as evidence."""
    rows = [
        {
            "source": "run/history.json",
            "where": "gate",
            "gate_effective": [0.0001] * 12,
            "gate_raw": [0.0001] * 12,
            "abs_mean": 0.0001,
            "abs_max": 0.0001,
        }
    ]
    text = describe(rows).lower()
    assert "0.000100" in text
    for verdict in ("closed", "open", "muted", "fail", "pass", "too small"):
        assert verdict not in text


def test_an_empty_sweep_says_so():
    assert describe([]) == "no gate records found"
