"""LP-relaxation solver: solve the continuous set-partition relaxation, then
round to a feasible integer assignment.

Fast (no branch-and-bound). The LP optimum is a lower bound on cost; rounding by
descending LP value with a greedy repair yields a strong feasible solution and,
on easy instances, the optimum. Useful as a quick high-quality strategy and to
report an optimality bound to the agent.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..problem import Problem
from ..solution import Solution
from ..columns import build_columns, to_matrix, Column
from .base import Solver


class LpSolver(Solver):
    name = "lp"

    def __init__(self, max_backup: int = 4, topk_single: int = 14,
                 subset_pool: int = 8, max_pairs: int = 80,
                 column_cap: int = 8000):
        self.max_backup = max_backup
        self.topk_single = topk_single
        self.subset_pool = subset_pool
        self.max_pairs = max_pairs
        self.column_cap = column_cap
        self.bound: float = float("nan")
        self._pool: Optional[List[Column]] = None  # agent may pre-build & share

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        try:
            import numpy as np
            from scipy.optimize import linprog
        except Exception:
            return self._finish(problem, [], t0)

        cols: List[Column] = self._pool if self._pool is not None else build_columns(
            problem, self.max_backup, self.topk_single,
            self.subset_pool, self.max_pairs)
        if not cols:
            return self._finish(problem, [], t0)
        if len(cols) > self.column_cap:
            cols = sorted(cols, key=lambda c: c.saving, reverse=True)[:self.column_cap]

        n = len(cols)
        A, tasks, cours = to_matrix(cols)
        cobj = np.array([-c.saving for c in cols])
        res = linprog(cobj, A_ub=A, b_ub=np.ones(len(tasks) + len(cours)),
                      bounds=(0, 1), method="highs")
        if res.x is None:
            return self._finish(problem, [], t0)
        self.bound = 100.0 * problem.task_count + res.fun  # LP cost lower bound

        # round: take columns by descending LP value, greedily keep feasible ones
        order = sorted(range(n), key=lambda j: (-res.x[j], -cols[j].saving))
        used_tasks = set()
        used_couriers = set()
        groups = []
        for j in order:
            if res.x[j] < 1e-6:
                break
            c = cols[j]
            if any(t in used_tasks for t in c.task_ids):
                continue
            if any(k in used_couriers for k in c.couriers):
                continue
            groups.append((c.task_key, list(c.couriers)))
            used_tasks.update(c.task_ids)
            used_couriers.update(c.couriers)

        # cover any tasks the rounding missed with the best remaining single column
        for c in sorted(cols, key=lambda c: c.cost):
            if len(c.task_ids) != 1:
                continue
            t = c.task_ids[0]
            if t in used_tasks or c.couriers[0] in used_couriers:
                continue
            groups.append((c.task_key, list(c.couriers)))
            used_tasks.add(t)
            used_couriers.add(c.couriers[0])

        sol = self._finish(problem, groups, t0)
        sol.bound = self.bound
        return sol
