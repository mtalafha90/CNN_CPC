from __future__ import annotations

from . import b21_training as training
from .b16_v2_report_ssl import (
    B16_V2_REPORT_EXPERIMENT,
    B16_V2_REPORT_OBJECTIVE,
    B16_V2_REPORT_VARIANT,
    load_b16_v2_report_encoder,
)


def activate_safe_encoder() -> None:
    training.load_b16_report_encoder = load_b16_v2_report_encoder
    training.B16_REPORT_SSL_VARIANT = B16_V2_REPORT_VARIANT
    training.B16_REPORT_SSL_EXPERIMENT = B16_V2_REPORT_EXPERIMENT
    training.B16_REPORT_SSL_OBJECTIVE = B16_V2_REPORT_OBJECTIVE


def main_control() -> None:
    activate_safe_encoder()
    training.main_control()


def main() -> None:
    activate_safe_encoder()
    training.main()


if __name__ == "__main__":
    main()
