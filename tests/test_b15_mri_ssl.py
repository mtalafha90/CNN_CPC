"""Contract tests for B15 MRI-domain SSL and matched weak-v2 downstream."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from rsna_knee.b15_downstream import require_v2_downstream_contract
from rsna_knee.b15_ssl import (
    WEAK_V2_MANIFEST_SHA256,
    _variable_ssl_examples,
    b15_ssl_study_pool,
)
from rsna_knee.constants import TARGETS


def test_frozen_v2_sha_is_exact():
    assert WEAK_V2_MANIFEST_SHA256 == (
        "1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1"
    )


def test_variable_ssl_examples_use_all_present_series_and_two_positions():
    volumes = torch.randn(2, 3, 5, 3, 8, 8)
    present = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    x, study_ids = _variable_ssl_examples(volumes, present, positions_per_series=2)
    assert x.shape == (10, 3, 8, 8)
    assert int((study_ids == 0).sum()) == 4
    assert int((study_ids == 1).sum()) == 6


def test_b15_ssl_pool_excludes_all_gold_and_v2_holdout():
    n = 4407
    uids = [f"s{i:04d}" for i in range(n)]
    frame = pd.DataFrame({"StudyInstanceUID": uids, "Report": ["r"] * n})
    for target in TARGETS:
        frame[target] = np.nan
    frame.loc[:57, TARGETS[0]] = 1.0
    non_gold = uids[58:]
    holdout = set(non_gold[:623])
    manifest = pd.DataFrame(
        {
            "StudyInstanceUID": non_gold,
            "report_group": [f"g{i}" for i in range(len(non_gold))],
            "split": ["holdout" if uid in holdout else "train" for uid in non_gold],
        }
    )
    pool, stats = b15_ssl_study_pool(frame, manifest)
    assert len(pool) == 3726
    assert not holdout.intersection(pool)
    assert not set(uids[:58]).intersection(pool)
    assert stats["excluded_gold_studies"] == 58
    assert stats["excluded_v2_holdout_studies"] == 623


def test_downstream_config_freezes_b13_recipe():
    config = yaml.safe_load(Path("configs/b15_mri_ssl.yaml").read_text())
    require_v2_downstream_contract(config)


def test_downstream_contract_rejects_architecture_drift():
    config = yaml.safe_load(Path("configs/b15_mri_ssl.yaml").read_text())
    config["b7_n_slices"] = 24
    with pytest.raises(ValueError, match="b7_n_slices"):
        require_v2_downstream_contract(config)
