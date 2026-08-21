"""B35 Phase-A training entrypoint with exact B34-matched encoder batching."""
from __future__ import annotations

from . import b35_training as _impl
from .b35_exact_batch import B35TargetSpatialResidualExactBatch

# The preserved training implementation resolves this symbol at runtime inside
# train_b35(). Rebinding it keeps every supervision/optimization/audit contract
# unchanged while replacing only the numerically sensitive encoder chunking.
_impl.B35TargetSpatialResidual = B35TargetSpatialResidualExactBatch


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()
