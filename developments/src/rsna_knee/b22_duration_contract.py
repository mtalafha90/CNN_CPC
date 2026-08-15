from __future__ import annotations

import numpy as np

from .b21_contract import require_b21_contract
from .b22_duration_protocol import B22_CROP_FRACTION, B22_EPOCHS, B22_SCHEDULER_HORIZON


def require_b22_duration_contract(config: dict) -> float:
    if int(config.get("b7_epochs", B22_EPOCHS)) != B22_EPOCHS:
        raise ValueError(f"B22 freezes b7_epochs={B22_EPOCHS}")
    if int(config.get("b22_scheduler_horizon", B22_SCHEDULER_HORIZON)) != B22_SCHEDULER_HORIZON:
        raise ValueError(f"B22 freezes scheduler horizon={B22_SCHEDULER_HORIZON}")
    surrogate = dict(config)
    surrogate["b7_epochs"] = 2
    fraction = require_b21_contract(surrogate)
    if not np.isclose(fraction, B22_CROP_FRACTION, atol=1e-12, rtol=0):
        raise ValueError("B22 freezes crop fraction at 0.90")
    return float(fraction)
