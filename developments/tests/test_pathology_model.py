import torch

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.pathology_model import PathologyAwareMILNet, default_target_stream_priors
from rsna_knee.pathology_training import _b3_model_spec, _policy_payload


def _small_b3(n_slices=2):
    return PathologyAwareMILNet(
        n_streams=len(DUAL_STREAMS),
        n_slices=n_slices,
        pretrained_weights=False,
        normalize_input=False,
        encoder_batch_size=2,
        gradient_checkpointing=False,
        dropout=0.0,
    ).eval()


def test_default_priors_are_positive_and_normalized():
    priors = default_target_stream_priors()
    assert priors.shape == (len(TARGETS), len(DUAL_STREAMS))
    assert torch.all(priors > 0)
    assert torch.allclose(priors.sum(dim=1), torch.ones(len(TARGETS)), atol=1e-6)


def test_pathology_aware_forward_and_attention_are_finite():
    model = _small_b3(2)
    x = torch.rand(1, 6, 2, 3, 64, 64)
    present = torch.tensor([[1, 1, 1, 0, 1, 0]], dtype=torch.float32)
    with torch.no_grad():
        logits, attention = model(x, present, return_attention=True)
    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()
    assert attention["stream_attention"].shape == (1, len(TARGETS), len(DUAL_STREAMS))
    assert torch.isfinite(attention["stream_attention"]).all()
    assert torch.allclose(
        attention["stream_attention"].sum(dim=-1),
        torch.ones(1, len(TARGETS)),
        atol=1e-5,
    )
    assert torch.all(attention["stream_attention"][:, :, 3] < 1e-6)
    assert torch.all(attention["stream_attention"][:, :, 5] < 1e-6)


def test_pathology_aware_allows_empty_study_without_nan():
    model = _small_b3(1)
    with torch.no_grad():
        logits = model(torch.zeros(1, 6, 1, 3, 64, 64), torch.zeros(1, 6))
    assert logits.shape == (1, len(TARGETS))
    assert torch.isfinite(logits).all()


def test_b3_spec_has_distinct_architecture_and_no_transformer_contract():
    spec = _b3_model_spec({"n_slices": 16, "image_size": 224})
    assert spec["architecture"] == "pathology_aware_stream_mil_v1"
    assert "transformer_layers" not in spec
    assert spec["prior_strength"] == 1.0


def test_b3_policy_declares_soft_not_hard_priors():
    payload = _policy_payload(
        {
            "ssl_encoder_checkpoint": "/tmp/ssl.pt",
            "ssl_checkpoint_source": "competition_training_data",
            "lr": 1e-4,
        }
    )
    assert payload["hard_stream_masks"] is False
    assert payload["soft_target_stream_priors"] is True
    assert payload["global_mri_transformer"] is False
    assert payload["pathology_interaction_transformer"] is False
