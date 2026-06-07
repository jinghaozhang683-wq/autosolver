"""Feature extraction helpers for learning logs."""

from __future__ import annotations

from typing import Dict, Optional

from ..problem import Problem
from ..solution import Solution


def solution_state(problem: Problem, incumbent: Optional[Solution],
                   remaining_time: float, num_moves_tried: int,
                   last_move: str = "", last_gain: float = 0.0) -> Dict[str, object]:
    """Summarise the online search state before choosing/running a move."""
    if incumbent is not None and incumbent.feasible:
        used_couriers = set(c for _, cs in incumbent.groups for c in cs)
        current_best = incumbent.score
        assigned_ratio = incumbent.assigned_count / max(1, problem.task_count)
        used_courier_ratio = len(used_couriers) / max(1, problem.courier_count)
    else:
        current_best = None
        assigned_ratio = 0.0
        used_courier_ratio = 0.0
    return {
        "remaining_time": round(max(0.0, remaining_time), 4),
        "current_best": current_best,
        "assigned_ratio": round(assigned_ratio, 4),
        "used_courier_ratio": round(used_courier_ratio, 4),
        "num_moves_tried": num_moves_tried,
        "last_move": last_move,
        "last_gain": round(last_gain, 6),
    }

