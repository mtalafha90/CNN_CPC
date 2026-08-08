from __future__ import annotations

from pathlib import Path


def validate_competition_config(config: dict, *, purpose: str) -> None:
    """Validate the conservative production contract used for Kaggle runs.

    The default policy intentionally assumes the strictest safe interpretation
    until competition-specific allowances are explicitly verified: one GPU,
    Internet-independent execution, runtime budget strictly below 9 h, and no
    external pretrained weights. Self-produced checkpoints/SSL weights trained
    only from competition data remain allowed.
    """
    if not bool(config.get("competition_mode", True)):
        return

    budget = float(config.get("runtime_budget_hours", 8.5))
    if not 0 < budget < 9.0:
        raise ValueError("competition_mode requires runtime_budget_hours strictly below 9")

    if bool(config.get("pretrained", False)) and not bool(config.get("allow_external_pretrained", False)):
        raise ValueError(
            "competition_mode forbids external pretrained weights by default; "
            "set pretrained:false unless the competition-specific rules have been explicitly verified"
        )

    if int(config.get("requested_gpus", 1)) != 1:
        raise ValueError("production competition mode is single-GPU only")

    if purpose == "infer":
        expected = str(config.get("submission_filename", "submission.csv"))
        if Path(expected).name != "submission.csv":
            raise ValueError("competition submission filename must be submission.csv")


def validate_submission_path(path: str | Path, config: dict) -> None:
    if bool(config.get("competition_mode", True)) and Path(path).name != "submission.csv":
        raise ValueError("competition_mode requires the output file to be named submission.csv")
