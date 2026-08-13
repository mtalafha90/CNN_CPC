from __future__ import annotations

import numpy as np

from .b21_protocol import B21_FIXED_EPOCHS, require_b21_crop_fraction


def require_b21_contract(config: dict) -> float:
    expected_int = {
        "seed": 2026,
        "requested_gpus": 1,
        "b7_n_slices": 16,
        "b7_image_size": 224,
        "b7_triplet_gap": 1,
        "b7_batch_size": 2,
        "b7_encoder_batch_size": 24,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_pathology_layers": 1,
        "b12_1_series_pool_heads": 8,
        "b7_epochs": B21_FIXED_EPOCHS,
        "b7_eval_batch_size": 2,
        "b7_n_bootstrap": 5000,
    }
    for key, expected in expected_int.items():
        if int(config.get(key, expected)) != expected:
            raise ValueError(f"B21 freezes {key}={expected}")

    expected_float = {
        "b7_dropout": 0.25,
        "b7_transformer_ff_mult": 2.0,
        "b7_encoder_lr": 0.0,
        "b7_head_lr": 0.0001,
        "b7_min_lr": 0.000001,
        "b7_weight_decay": 0.0001,
        "b7_grad_clip": 1.0,
        "b7_noise_std": 0.02,
        "b7_slice_dropout": 0.08,
        "b7_center_jitter": 2.0,
        "b7_rotation_deg": 5.0,
        "b7_translate_frac": 0.03,
        "b7_scale_jitter": 0.05,
        "b7_gamma_jitter": 0.12,
        "b7_bias_field_strength": 0.08,
        "b7_min_confidence": 0.75,
        "b7_positive_target": 0.85,
        "b7_negative_target": 0.05,
        "b7_positive_weight": 0.50,
        "b7_negative_weight": 1.00,
        "b17_label_smoothing": 0.0,
    }
    for key, expected in expected_float.items():
        value = float(config.get(key, expected))
        if not np.isclose(value, expected, atol=1e-12, rtol=0):
            raise ValueError(f"B21 freezes {key}={expected}")

    if str(config.get("b17_robust_loss", "none")) != "none":
        raise ValueError("B21 freezes robust loss to none")
    if bool(config.get("b18_expert_selection", False)):
        raise ValueError("B21 disables expert checkpoint selection")
    if bool(config.get("b12_use_physical_scale", False)):
        raise ValueError("B21 keeps physical scale disabled")
    return require_b21_crop_fraction(config)
