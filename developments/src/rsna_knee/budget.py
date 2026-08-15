from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RuntimeBudget:
    """Wall-clock guard for Kaggle code-competition runs.

    The production config uses a budget below the platform ceiling and reserves
    the final minutes for serialization/cleanup. ``work_deadline_monotonic`` is
    an absolute stop point for scheduling new GPU work, so direct function calls
    receive the same protection as the CLI.
    """

    max_hours: float = 8.5
    reserve_minutes: float = 10.0
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not 0 < float(self.max_hours) < 9.0:
            raise ValueError("runtime budget must be >0 and strictly <9 hours")
        if float(self.reserve_minutes) < 0:
            raise ValueError("reserve_minutes must be >=0")
        if float(self.reserve_minutes) * 60.0 >= float(self.max_hours) * 3600.0:
            raise ValueError("reserve_minutes must be shorter than the runtime budget")

    @property
    def max_seconds(self) -> float:
        return float(self.max_hours) * 3600.0

    @property
    def reserve_seconds(self) -> float:
        return float(self.reserve_minutes) * 60.0

    @property
    def hard_deadline_monotonic(self) -> float:
        return float(self.started) + self.max_seconds

    @property
    def work_deadline_monotonic(self) -> float:
        return self.hard_deadline_monotonic - self.reserve_seconds

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    @property
    def remaining_work_seconds(self) -> float:
        return max(0.0, self.work_deadline_monotonic - time.monotonic())

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

    def require_components(self, components: dict[str, float], *, label: str) -> float:
        """Require enough time for a named collection of remaining tasks."""
        total = sum(max(0.0, float(value)) for value in components.values())
        self.require(total, label=label)
        return total

    def to_dict(self) -> dict:
        return {
            "max_hours": float(self.max_hours),
            "reserve_minutes": float(self.reserve_minutes),
            "elapsed_seconds": float(self.elapsed_seconds),
            "remaining_seconds": float(self.remaining_seconds),
            "remaining_work_seconds": float(self.remaining_work_seconds),
            "work_deadline_monotonic": float(self.work_deadline_monotonic),
        }
