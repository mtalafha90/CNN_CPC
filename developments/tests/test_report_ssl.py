from __future__ import annotations

import numpy as np
import torch

from rsna_knee.report_ssl import (
    _aggregate_study_features,
    _report_alignment_losses,
    _update_report_queue,
    fit_report_semantics,
)


def test_fit_report_semantics_is_finite_normalized_and_grouped():
    reports = [
        "ACL tear with joint effusion",
        "ACL tear with joint effusion",
        "Medial meniscus tear",
        "Tricompartmental osteoarthritis",
        "No acute fracture",
        "Baker cyst and synovitis",
    ]
    vectors, groups, vectorizer, svd, stats = fit_report_semantics(
        reports,
        requested_dim=4,
        max_features=100,
        min_df=1,
        seed=7,
    )
    assert vectors.shape == (6, 4)
    assert groups.shape == (6,)
    assert groups[0] == groups[1]
    assert stats["duplicate_report_rows"] == 1
    assert stats["reports"] == 6
    assert stats["external_text_model"] is False
    assert len(vectorizer.vocabulary_) > 0
    assert svd.n_components == 4
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_fit_report_semantics_handles_empty_report_token():
    reports = ["", "ACL tear", "Normal knee", "Joint effusion"]
    vectors, groups, _, _, stats = fit_report_semantics(
        reports,
        requested_dim=2,
        max_features=50,
        min_df=1,
        seed=1,
    )
    assert vectors.shape == (4, 2)
    assert groups.shape == (4,)
    assert stats["empty_reports"] == 1
    assert np.isfinite(vectors).all()


def test_aggregate_study_features_means_active_examples():
    feat = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 5.0],
            [10.0, 20.0],
        ]
    )
    study_ids = torch.tensor([0, 0, 2], dtype=torch.long)
    pooled, valid = _aggregate_study_features(feat, study_ids, batch_size=3)
    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(pooled[2], torch.tensor([10.0, 20.0]))
    assert valid.tolist() == [True, False, True]


def test_report_alignment_prefers_matching_embeddings():
    report = torch.eye(3, dtype=torch.float32)
    groups = torch.tensor([0, 1, 2], dtype=torch.long)
    nce_good, cos_good = _report_alignment_losses(
        report,
        report,
        groups,
        temperature=0.1,
    )
    shuffled = report[[1, 2, 0]]
    nce_bad, cos_bad = _report_alignment_losses(
        shuffled,
        report,
        groups,
        temperature=0.1,
    )
    assert torch.isfinite(nce_good)
    assert torch.isfinite(cos_good)
    assert nce_good < nce_bad
    assert cos_good < cos_bad


def test_report_alignment_masks_duplicate_report_negatives():
    image = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    report = image.clone()
    groups = torch.tensor([5, 5], dtype=torch.long)
    nce, cosine = _report_alignment_losses(
        image,
        report,
        groups,
        temperature=0.1,
    )
    # Each duplicate counterpart is masked, leaving the diagonal positive.
    assert torch.isfinite(nce)
    assert float(nce) < 1e-4
    assert float(cosine) < 1e-6


def test_report_queue_respects_capacity_and_normalizes():
    qz, qg = _update_report_queue(
        None,
        None,
        torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([1, 2], dtype=torch.long),
        capacity=3,
    )
    qz, qg = _update_report_queue(
        qz,
        qg,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        torch.tensor([3, 4], dtype=torch.long),
        capacity=3,
    )
    assert qz.shape == (3, 2)
    assert qg.tolist() == [2, 3, 4]
    torch.testing.assert_close(torch.linalg.vector_norm(qz, dim=1), torch.ones(3))


def test_zero_capacity_disables_report_queue():
    qz, qg = _update_report_queue(
        None,
        None,
        torch.ones((2, 3)),
        torch.tensor([0, 1]),
        capacity=0,
    )
    assert qz is None
    assert qg is None
