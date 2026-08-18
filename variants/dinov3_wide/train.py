"""Train the working model with a wide DINOv3 encoder.

This reuses the frozen training loop unchanged -- same supervision, population,
series exposure, crop, augmentation, optimiser, seeds and fixed endpoint. Only
the model construction is redirected, so the run differs from the supported one
in exactly the encoder and the head width that follows from it.

The redirection is done by substituting three names inside the frozen trainer's
namespace for the duration of the call, rather than by copying the training
loop. A copy would drift; a substitution cannot, because there is only one loop.

This is an experiment. It is deliberately outside the supported interface, and
its results are not comparable to a frozen-width run without saying so: the head
is wider too, so encoder pretraining and head capacity move together.
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

from model._implementation import ensure_developments_source, read_config, run_directory

from .encoder import DINOV3_MODELS
from .model import _contracts_scaled_to, build_wide_model, wide_model_spec

ensure_developments_source()

from rsna_knee import phase9_matched_supervision_training as trainer  # noqa: E402

SUPERVISION_SURFACES = {"latin-script": "control", "all-script": "candidate"}


@contextlib.contextmanager
def _trainer_builds_wide(variant: str):
    """Redirect the frozen trainer's model construction to the wide variant."""
    width = DINOV3_MODELS[variant][1]
    saved = {
        name: getattr(trainer, name)
        for name in ("b34_model_spec", "build_b34_model", "attach_dinov3_encoder")
    }

    def spec_factory(config, *, normalize_input: bool):
        return wide_model_spec(config, variant, normalize_input=normalize_input)

    def model_factory(spec, *, pretrained_weights: bool = False):
        # The encoder arrives already carrying DINOv3 weights.
        return build_wide_model(spec, pretrained_weights=True)

    def already_attached(model, *, variant=None, pretrained_weights=True):
        return model.encoder

    trainer.b34_model_spec = spec_factory
    trainer.build_b34_model = model_factory
    trainer.attach_dinov3_encoder = already_attached
    try:
        with _contracts_scaled_to(width):
            yield
    finally:
        for name, original in saved.items():
            setattr(trainer, name, original)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with a wide DINOv3 encoder")
    parser.add_argument("--supervision", choices=tuple(SUPERVISION_SURFACES), required=True)
    parser.add_argument(
        "--dinov3-variant",
        choices=tuple(DINOV3_MODELS),
        default="base",
        help="tiny and small keep the frozen 768-d head; base and large widen it",
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latin-script-labels", required=True)
    parser.add_argument("--all-script-labels", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())

    width = DINOV3_MODELS[args.dinov3_variant][1]
    print(
        f"[wide] variant={args.dinov3_variant} encoder_width={width} "
        f"supervision={args.supervision}"
    )
    if width != 768:
        print(
            "[wide] the head is built at this width too, so encoder pretraining "
            "and head capacity both differ from the frozen model"
        )

    with _trainer_builds_wide(args.dinov3_variant):
        checkpoint = trainer.train_phase9_arm(
            config,
            arm=SUPERVISION_SURFACES[args.supervision],
            b6_root=args.latin_script_labels,
            phase8_root=args.all_script_labels,
            series_policy_path=args.series_policy,
            report_ssl_checkpoint="",
            out_root=run_directory(args.experiment, "train"),
            out_dirname=args.supervision,
            encoder_source="dinov3",
        )
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
