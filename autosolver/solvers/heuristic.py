"""Heuristic solver: the pure-Python beam-search + local-search engine.

Wraps the standalone `solver.py` (the no-third-party-library engine tuned for
the judge). It is the strongest strategy on the largest instances, where exact
column methods run out of time. Self-contained (no external deps), so it is also
the safe fallback if scipy / ortools are unavailable.
"""

from __future__ import annotations

import time
from typing import Optional

from ..problem import Problem
from ..solution import Solution
from .base import Solver
from .greedy import GreedySolver


class HeuristicSolver(Solver):
    name = "heuristic"

    def __init__(self, time_budget_override: Optional[float] = None):
        self._engine = None
        self._fallback = GreedySolver()
        self.time_budget_override = time_budget_override
        self._warned = False   # warn once per process about a missing engine

    def _load(self):
        if self._engine is None:
            import importlib
            self._engine = importlib.import_module("solver")
        return self._engine

    def _warn_degraded(self, exc: Exception) -> None:
        """Make a missing-engine fallback *loud*: if solver.py is not packaged,
        the heuristic silently behaving like greedy would quietly cost score on
        the judge. Emit a one-time stderr warning so it is visible."""
        if self._warned:
            return
        self._warned = True
        import sys
        sys.stderr.write(
            "[autosolver] WARNING: heuristic engine 'solver.py' could not be "
            "imported (%s); falling back to GREEDY. Solution quality is "
            "DEGRADED -- ship solver.py on the import path to restore it.\n"
            % exc)

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        try:
            engine = self._load()
        except Exception as exc:
            self._warn_degraded(exc)
            sol = self._fallback.solve(problem, time_budget, incumbent)
            sol.strategy = self.name
            sol.degraded = True
            sol.solve_time = time.time() - t0
            return sol
        # rebuild the raw input text the engine expects
        text = self._to_text(problem)
        budget = self.time_budget_override or max(0.2, time_budget - 0.05)
        try:
            groups = engine.solve_with_time_limit(text, budget)
        except Exception:
            try:
                groups = engine.solve(text)
            except Exception:
                groups = []
        return self._finish(problem, [(tk, list(cs)) for tk, cs in groups], t0)

    def _to_text(self, problem: Problem) -> str:
        lines = ["task_id_list\tstation_id\tscore\twillingness"]
        for c in problem.candidates:
            lines.append("%s\t%s\t%.6f\t%.6f" % (
                c.task_key, c.courier_id, c.score, c.willingness))
        return "\n".join(lines)
