"""B35 expert diagnostic using the exact B34-matched encoder batching fix."""
from __future__ import annotations

from . import b35_eval as _impl
from .b35_exact_batch import B35TargetSpatialResidualExactBatch

# load_b35_checkpoint() resolves this module-global class when reconstructing the
# candidate, so the evaluation uses the same numerically matched implementation
# as training without changing the checkpoint schema.
_impl.B35TargetSpatialResidual = B35TargetSpatialResidualExactBatch


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()
