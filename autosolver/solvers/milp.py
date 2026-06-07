"""Exact set-partition solver via scipy.optimize.milp (HiGHS branch-and-bound).

Reliable on tens of thousands of columns; reaches the true optimum on small and
medium instances within a few seconds. On very large instances it returns the
best integer solution found within the time budget.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..problem import Problem
from ..solution import Solution
from ..columns import build_columns, to_matrix, Column
from .base import Solver


class MilpSolver(Solver):
    name = "milp"

    def __init__(self, max_backup: int = 4, topk_single: int = 18,
                 subset_pool: int = 13, max_pairs: int = 150,
                 column_cap: int = 40000):
        self.max_backup = max_backup
        self.topk_single = topk_single
        self.subset_pool = subset_pool
        self.max_pairs = max_pairs
        self.column_cap = column_cap
        self.column_scorer = None
        self.backup_scorer = None
        self._pool: Optional[List[Column]] = None  # agent may pre-build & share

    def _column_pool_complete(self, problem: Problem, cols: List[Column],
                              capped: bool) -> bool:
        """Return True only when the generated pool enumerates every column.

        HiGHS can prove optimality for the variables it sees. That is a global
        certificate only if column generation did not prune bundles, couriers,
        backup-set sizes, or the final pool.
        """
        if capped:
            return False
        pair_count = sum(1 for cands in problem.by_task_key.values()
                         if cands and cands[0].task_count >= 2)
        if self.max_pairs < pair_count:
            return False
        for cands in problem.by_task_key.values():
            n = len(cands)
            if self.topk_single < n:
                return False
            if self.subset_pool < n:
                return False
            if self.max_backup < n:
                return False
        return True

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        try:
            import numpy as np
            from scipy.optimize import milp, LinearConstraint, Bounds
        except Exception:
            return self._finish(problem, [], t0)

        cols: List[Column] = self._pool if self._pool is not None else build_columns(
            problem, self.max_backup, self.topk_single,
            self.subset_pool, self.max_pairs,
            column_scorer=self.column_scorer,
            backup_scorer=self.backup_scorer)
        if not cols:
            return self._finish(problem, [], t0)
        capped = False
        if len(cols) > self.column_cap:
            cols = sorted(cols, key=lambda c: c.saving, reverse=True)[:self.column_cap]
            capped = True

        n = len(cols)
        A, tasks, cours = to_matrix(cols)
        cobj = np.array([-c.saving for c in cols])  # maximise saving
        con = LinearConstraint(A, -np.inf, 1)
        # HiGHS treats time_limit as a soft bound and matrix setup adds time, so
        # leave a margin to avoid overrunning the agent's budget.
        remaining = max(0.2, time_budget - (time.time() - t0) - 0.8)
        try:
            res = milp(c=cobj, constraints=con, integrality=np.ones(n),
                       bounds=Bounds(0, 1),
                       options={"time_limit": remaining, "mip_rel_gap": 0.0})
        except Exception:
            return self._finish(problem, [], t0)

        groups = []
        if res.x is not None:
            for j in range(n):
                if res.x[j] > 0.5:
                    groups.append((cols[j].task_key, list(cols[j].couriers)))
        sol = self._finish(problem, groups, t0)
        solved_to_optimality = (getattr(res, "status", 1) == 0)
        complete_pool = self._column_pool_complete(problem, cols, capped)
        sol.restricted_optimal = solved_to_optimality
        sol.optimal = solved_to_optimality and complete_pool
        sol.optimal_scope = (
            "global" if sol.optimal
            else "restricted" if sol.restricted_optimal
            else "none"
        )
        return sol
