import numpy as np

from rsna_knee.b20_explain import _cam_metrics, _mass_mask


def test_mass_mask_contains_requested_cam_mass_without_empty_mask():
    cam = np.array([[0.9, 0.6], [0.3, 0.2]], dtype=float)
    mask = _mass_mask(cam, 0.75)
    assert mask.dtype == bool
    assert mask.any()
    retained = float(cam[mask].sum() / cam.sum())
    assert retained >= 0.75


def test_cam_metrics_peak_and_center_of_mass_are_finite():
    cam = np.zeros((5, 5), dtype=float)
    cam[1, 3] = 1.0
    cam[1, 2] = 0.5
    metrics = _cam_metrics(cam)
    assert metrics["cam_peak_y"] == 1
    assert metrics["cam_peak_x"] == 3
    assert np.isfinite(metrics["cam_center_of_mass_y_norm"])
    assert np.isfinite(metrics["cam_center_of_mass_x_norm"])
    assert 0.0 <= metrics["cam_normalized_entropy"] <= 1.0
