from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RuntimeBudget:
    """Wall-clock guard for Kaggle code-competition runs.

    Kaggle GPU submissions must finish before the platform limit. The production
    config uses an 8.5 h budget so the program has a safety margin below 9 h.
    The guard never starts a new optional unit of work when the remaining time is
    insufficient for the supplied estimate plus the reserve.
    """

    max_hours: float = 8.5
    reserve_minutes: float = 10.0
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not 0 < float(self.max_hours) < 9.0:
            raise ValueError("runtime budget must be >0 and strictly <9 hours")
        if float(self.reserve_minutes) < 0:
            raise ValueError("reserve_minutes must be >=0")

    @property
    def max_seconds(self) -> float:
        return float(self.max_hours) * 3600.0

    @property
    def reserve_seconds(self) -> float:
        return float(self.reserve_minutes) * 60.0

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    def can_start(self, estimated_seconds: float = 0.0, *, extra_reserve_seconds: float = 0.0) -> bool:
        required = max(0.0, float(estimated_seconds)) + self.reserve_seconds + max(0.0, float(extra_reserve_seconds))
        return self.remaining_seconds > required

    def require(self, estimated_seconds: float = 0.0, *, label: str = "work") -> None:
        if not self.can_start(estimated_seconds):
            raise RuntimeError(
                f"runtime budget would be exceeded before {label}: "
                f"remaining={self.remaining_seconds/60:.1f} min, "
                f"estimate={float(estimated_seconds)/60:.1f} min, "
                f"reserve={self.reserve_minutes:.1f} min"
            )

    def to_dict(self) -> dict:
        return {
            "max_hours": float(self.max_hours),
            "reserve_minutes": float(self.reserve_minutes),
            "elapsed_seconds": float(self.elapsed_seconds),
            "remaining_seconds": float(self.remaining_seconds),
        }
