"""Score the B51 endpoint on the reused Expert-58 surface.

The computation is B42's, unchanged and deliberately so. B50's model subclasses
B42's and alters only `requires_grad`, which a forward pass ignores, so the two
share one inference recipe: the same 90% native crop, constant-area native-aspect
resize, reflection padding, ragged per-series encoding, three centre offsets,
6x6 grid and top-k=8. Re-deriving any of that here would risk measuring a
different thing and calling it a comparison.

What this module adds is a name. Run through `evaluate_b42` directly, a B51
checkpoint produces a folder of files called `b42_combined_predictions.csv`,
result keys called `b42_minus_b37_combined_macro`, and a note saying B42 must
not be tuned from them. Months later nothing in that folder would say which
weights made the numbers. So the label travels with the evaluation, and the
result records the checkpoint's own experiment, version and conversion origin.

Read the result with its resolution in mind. The Expert-58 surface is 58
studies, which resolves to roughly +/-0.03. B50 measured the adapted hierarchy
at +0.011221 on 548 unseen-scanner studies. This diagnostic therefore cannot
confirm or refute that effect, and was never able to: B50's own Expert-58 delta
was -0.002432, which is inconclusive rather than negative. Its real use is
confirming the converted checkpoint reconstructs and runs, and catching a large
regression that would mean something is actually broken.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b42_constant_area_aspect_sparse_eval import evaluate_b42
from .b51_full_population_training import B51_RUN_ROOT
from .b51_submission_dualgpu_fast import require_converted_b51

B51_EXPERT58_ROOT = f"{B51_RUN_ROOT}/expert58"
B51_EVAL_LABEL = "b51"

# Expert-58 is 58 studies. B50's measured effect is about a third of what that
# can resolve, so a small delta here carries no information either way.
B51_EXPERT58_RESOLUTION = 0.03
B50_UNSEEN_SCANNER_DELTA = 0.011221


def evaluate_b51(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B51_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
) -> dict:
    """Run B42's Expert-58 evaluation against a converted B51 checkpoint."""
    identity = require_converted_b51(checkpoint)

    resolved_out = Path(out_root).resolve()
    print(f"[B51 eval] checkpoint sha256 {identity['sha256']}", flush=True)
    print(
        f"[B51 eval] trained on {identity['training_studies']} studies, "
        f"hierarchy lr scale {identity['hierarchy_lr_scale']}",
        flush=True,
    )
    print(f"[B51 eval] writing {resolved_out}", flush=True)
    print(
        f"[B51 eval] 58 studies resolve to about +/-{B51_EXPERT58_RESOLUTION:.2f}; "
        f"B50 measured +{B50_UNSEEN_SCANNER_DELTA:.6f}. A small delta here means "
        "nothing either way.",
        flush=True,
    )

    result = evaluate_b42(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_root=resolved_out,
        n_bootstrap=n_bootstrap,
        experiment_label=B51_EVAL_LABEL,
    )
    delta = result.get(f"{B51_EVAL_LABEL}_minus_b41_combined_macro")
    if delta is not None and abs(float(delta)) < B51_EXPERT58_RESOLUTION:
        print(
            f"[B51 eval] delta vs B41 is {float(delta):+.6f}, inside this "
            "surface's resolution: inconclusive, not evidence either way.",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the converted B51 checkpoint on the reused Expert-58 surface"
    )
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="the converted B51 checkpoint")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B51_EXPERT58_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    evaluate_b51(
        dict(config),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()


__all__ = ["B51_EXPERT58_ROOT", "B51_EVAL_LABEL", "evaluate_b51"]
