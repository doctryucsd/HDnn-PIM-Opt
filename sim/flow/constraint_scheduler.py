from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal


Constraints = Dict[str, float]


DEFAULT_START_CONSTRAINTS: Constraints = {
    # Start permissive: low accuracy lower bound, high upper bounds
    "accuracy": 0.0,      # lower bound
    "power": 1.0,         # upper bound
    "performance": 1.0,   # upper bound
    "area": 1.0,          # upper bound
}


def _interp_linear(start: float, end: float, p: float) -> float:
    return start + (end - start) * p


def _interp_exponential(start: float, end: float, p: float, k: float = 5.0) -> float:
    # Smooth exponential easing from start to end, p in [0,1]
    if p <= 0.0:
        return start
    if p >= 1.0:
        return end
    denom = 1.0 - pow(2.718281828459045, -k)
    factor = (1.0 - pow(2.718281828459045, -k * p)) / denom
    return start + (end - start) * factor


@dataclass
class ConstraintScheduler:
    schedule_type: Literal["static", "linear", "exponential"]
    end_constraints: Constraints
    start_constraints: Constraints = field(default_factory=lambda: dict(DEFAULT_START_CONSTRAINTS))
    exp_k: float = 5.0

    def get(self, step_idx: int, total_steps: int) -> Constraints:
        """
        Compute scheduled constraints for a given step.

        step_idx: 0-based index within the scheduled phase (after Sobol).
        total_steps: total number of scheduled steps (>= 1).
        """
        total_steps = max(int(total_steps), 1)
        # Map step_idx in [0, total_steps-1] to progress p in [0,1]
        if total_steps == 1:
            p = 1.0
        else:
            p = max(0.0, min(1.0, step_idx / float(total_steps - 1)))

        if self.schedule_type == "static":
            return dict(self.end_constraints)

        out: Constraints = {}
        for key in self.end_constraints.keys():
            s = float(self.start_constraints.get(key, self.end_constraints[key]))
            e = float(self.end_constraints[key])
            if self.schedule_type == "linear":
                out[key] = _interp_linear(s, e, p)
            elif self.schedule_type == "exponential":
                out[key] = _interp_exponential(s, e, p, self.exp_k)
            else:
                # Fallback to static if unknown
                out[key] = e
        return out


def make_constraint_scheduler(
    schedule_type: str, end_constraints: Constraints
) -> ConstraintScheduler:
    st = schedule_type.lower()
    if st not in {"static", "linear", "exponential"}:
        raise ValueError(f"Unknown threshold schedule type: {schedule_type}")
    return ConstraintScheduler(st, dict(end_constraints), dict(DEFAULT_START_CONSTRAINTS))
