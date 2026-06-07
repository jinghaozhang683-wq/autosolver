"""Exact set-partition solver via Google OR-Tools CP-SAT.

Model (binary x[c] for each column c):

    minimise  sum_c (cost_c * x_c) + 100 * (n_tasks - covered)
              = sum_c (cost_c - 100 * |tasks_c|) * x_c  + const
    s.t.      for each task t:    sum_{c covers t} x_c <= 1
              for each courier k: sum_{c uses  k} x_c <= 1

Because cost_c - 100*|tasks_c| = -saving_c, this is equivalently
*maximise total saving*. Uncovered tasks simply keep their 100 penalty.

CP-SAT works in integers, so costs are scaled to integers (x1000).
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..problem import Problem
from ..solution import Solution
from ..columns import build_columns, Column
from .base import Solver

_SCALE = 1000


class CpSatSolver(Solver):
    name = "cpsat"

    def __init__(self, max_backup: int = 4, topk_single: int = 12,
                 subset_pool: int = 7, max_pairs: int = 100, workers: int = 8,
                 column_cap: int = 8000):
        self.max_backup = max_backup
        self.topk_single = topk_single
        self.subset_pool = subset_pool
        self.max_pairs = max_pairs
        self.workers = workers
        self.column_cap = column_cap
        self._pool: Optional[List[Column]] = None  # agent may pre-build & share

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        try:
            from ortools.sat.python import cp_model
        except Exception:
            return self._finish(problem, [], t0)

        columns: List[Column] = self._pool if self._pool is not None else build_columns(
            problem, self.max_backup, self.topk_single,
            self.subset_pool, self.max_pairs)
        if not columns:
            return self._finish(problem, [], t0)
        # CP-SAT presolve struggles past a few thousand near-identical columns;
        # keep the highest-saving ones.
        if len(columns) > self.column_cap:
            columns = sorted(columns, key=lambda c: c.saving,
                             reverse=True)[:self.column_cap]

        model = cp_model.CpModel()
        x = [model.NewBoolVar("c%d" % i) for i in range(len(columns))]

        # task coverage <= 1
        task_cols = {}
        courier_cols = {}
        for i, col in enumerate(columns):
            for t in col.task_ids:
                task_cols.setdefault(t, []).append(i)
            for k in col.couriers:
                courier_cols.setdefault(k, []).append(i)
        for t, idxs in task_cols.items():
            model.Add(sum(x[i] for i in idxs) <= 1)
        for k, idxs in courier_cols.items():
            model.Add(sum(x[i] for i in idxs) <= 1)

        # minimise sum (cost - 100*|tasks|) = -saving ; maximise saving
        obj = [int(round(col.saving * _SCALE)) * x[i]
               for i, col in enumerate(columns)]
        model.Maximize(sum(obj))

        solver = cp_model.CpSolver()
        remaining = max(0.1, time_budget - (time.time() - t0))
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = self.workers
        # warm start from incumbent if provided
        if incumbent is not None and incumbent.feasible:
            self._hint(model, solver, x, columns, incumbent)

        status = solver.Solve(model)
        groups = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i, col in enumerate(columns):
                if solver.Value(x[i]) == 1:
                    groups.append((col.task_key, list(col.couriers)))

        sol = self._finish(problem, groups, t0)
        # CP-SAT can only handle a few thousand columns, so it runs on a tightly
        # capped pool. "OPTIMAL" here means optimal *over that restricted pool* --
        # not a global certificate -- so we do NOT claim global optimality (that
        # would let the agent stop early on a pool-restricted answer). The
        # generous-pool MILP engine is the one trusted to certify optimality.
        sol.restricted_optimal = (status == cp_model.OPTIMAL)
        sol.optimal = False
        sol.optimal_scope = "restricted" if sol.restricted_optimal else "none"
        sol.bound = solver.BestObjectiveBound() / _SCALE  # type: ignore[attr-defined]
        return sol

    def _hint(self, model, solver, x, columns, incumbent):
        # map incumbent groups -> matching columns to hint the solver
        want = set()
        for tk, cs in incumbent.groups:
            want.add((tk, tuple(sorted(cs))))
        for i, col in enumerate(columns):
            if (col.task_key, tuple(sorted(col.couriers))) in want:
                try:
                    model.AddHint(x[i], 1)
                except Exception:
                    pass
