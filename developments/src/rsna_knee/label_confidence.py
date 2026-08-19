"""Set how confident a parsed report is allowed to make the model.

The frozen B7 policy aims a report's "yes" at 0.85 and its "no" at 0.05. Those
numbers were chosen before anything measured them. Auditing the parser against
the 58 expert-labelled studies since put figures on it: when the parser says a
finding is present the expert agrees about 69% of the time, and when it says
absent, about 96%.

So a positive cell is trained towards a confidence a third higher than the
label deserves, while a negative cell is trained towards one slightly lower
than it deserves. Whether correcting that helps is an open question -- an
over-confident target is not automatically harmful, and the loss is weighted
as well as targeted -- but it is a question worth being able to ask.

The targets live in the exported supervision table, not in the config, because
they were written when the labels were. Rescaling therefore happens here, on
the loaded array, which is the only place the values actually exist.
"""
from __future__ import annotations

import numpy as np

# What the frozen export contains, and what a rescale replaces.
EXPORTED_POSITIVE_TARGET = 0.85
EXPORTED_NEGATIVE_TARGET = 0.05

# A cell counts as positive or negative by which side of this it falls, in the
# training loop and in the audit counts alike. Targets must stay on their own
# side of it or those counts quietly start describing something else.
DECISION_BOUNDARY = 0.5


def rescale_label_confidence(
    targets: np.ndarray, weights: np.ndarray, config: dict
) -> tuple[np.ndarray, dict]:
    """Move the supervised targets onto the configured confidences.

    Only cells that carry weight are touched, so unsupervised cells keep
    whatever placeholder the export gave them. Returns the new targets and a
    record of what happened, for the training payload.
    """
    positive_target = float(config.get("b7_positive_target", EXPORTED_POSITIVE_TARGET))
    negative_target = float(config.get("b7_negative_target", EXPORTED_NEGATIVE_TARGET))

    for name, value in (("positive", positive_target), ("negative", negative_target)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} target must lie between 0 and 1, not {value}")
    if positive_target <= DECISION_BOUNDARY:
        raise ValueError(
            f"positive target must stay above {DECISION_BOUNDARY}, not {positive_target}; "
            "below it, positive cells would be counted as negatives"
        )
    if negative_target >= DECISION_BOUNDARY:
        raise ValueError(
            f"negative target must stay below {DECISION_BOUNDARY}, not {negative_target}; "
            "above it, negative cells would be counted as positives"
        )

    changed = not (
        np.isclose(positive_target, EXPORTED_POSITIVE_TARGET)
        and np.isclose(negative_target, EXPORTED_NEGATIVE_TARGET)
    )
    record = {
        "changed": bool(changed),
        "positive_target": positive_target,
        "negative_target": negative_target,
        "exported_positive_target": EXPORTED_POSITIVE_TARGET,
        "exported_negative_target": EXPORTED_NEGATIVE_TARGET,
        "measured_positive_agreement": 0.690,
        "measured_negative_agreement": 0.964,
        "measurement_source": "tools.label_audit over the 58 expert-labelled studies",
    }
    if not changed:
        return targets, record

    rescaled = np.array(targets, dtype=targets.dtype, copy=True)
    supervised = weights > 0
    positives = supervised & (targets > DECISION_BOUNDARY)
    negatives = supervised & (targets < DECISION_BOUNDARY)
    rescaled[positives] = positive_target
    rescaled[negatives] = negative_target

    record["positive_cells_rescaled"] = int(positives.sum())
    record["negative_cells_rescaled"] = int(negatives.sum())
    return rescaled, record
