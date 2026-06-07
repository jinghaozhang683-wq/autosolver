"""Common solver interface.

Every strategy implements `solve(problem, time_budget, **kw) -> Solution`.
Solvers must respect `time_budget` (seconds) and always return a *feasible*
Solution (possibly empty) — never raise on a solvable instance.
"""

from __future__ import annotations

import time
from typing import Optional

from ..problem import Problem
from ..solution import Solution


class Solver:
    name: str = "base"

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        raise NotImplementedError

    # convenience: evaluate + tag strategy name
    def _finish(self, problem: Problem, groups, t0: float) -> Solution:
        sol = Solution(groups, strategy=self.name)
        sol.evaluate(problem)
        sol.solve_time = time.time() - t0  # type: ignore[attr-defined]
        return sol
