"""Greedy construction strategies.

Fast baseline: pick assignments (single couriers or pre-formed bundles) by
several priority orders, respecting task/courier uniqueness, then add backup
couriers where they lower the expected group cost. Always feasible, sub-second.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..problem import Problem, group_cost, UNASSIGNED_PENALTY, EPS, Candidate
from ..solution import Solution
from .base import Solver


def _saving(c: Candidate) -> float:
    return UNASSIGNED_PENALTY * c.task_count - c.cost


_ORDERS = [
    lambda c: (-_saving(c), c.cost / max(c.task_count, 1)),
    lambda c: (c.cost / max(c.task_count, 1), -_saving(c)),
    lambda c: (-c.willingness, c.cost / max(c.task_count, 1)),
    lambda c: (-c.task_count, -_saving(c), c.cost),
    lambda c: (-_saving(c) / max(c.task_count, 1), -c.willingness),
]


class GreedySolver(Solver):
    name = "greedy"

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        positive = [c for c in problem.candidates if _saving(c) > EPS]
        best_groups = None
        best = None
        for keyfn in _ORDERS:
            groups = self._construct(problem, sorted(positive, key=keyfn))
            sol = Solution(groups, strategy=self.name)
            sol.evaluate(problem)
            if sol.better_than(best):
                best = sol
                best_groups = groups
            if time.time() - t0 > time_budget:
                break
        if best is None:
            return self._finish(problem, [], t0)
        # add backups to the best primary selection
        best_groups = self._add_backups(problem, best_groups)
        return self._finish(problem, best_groups, t0)

    def _construct(self, problem, ordered):
        used_tasks = set()
        used_couriers = set()
        groups = []
        for c in ordered:
            if c.courier_id in used_couriers:
                continue
            if any(t in used_tasks for t in c.task_ids):
                continue
            groups.append((c.task_key, [c.courier_id]))
            used_couriers.add(c.courier_id)
            used_tasks.update(c.task_ids)
        return groups

    def _add_backups(self, problem, groups):
        used = set(cid for _, cs in groups for cid in cs)
        out = []
        for task_key, couriers in groups:
            cands = problem.by_task_key.get(task_key, [])
            tcount = problem.bundle_tasks(task_key).__len__()
            members = [(problem.info(task_key, cid)[0],
                        problem.info(task_key, cid)[1]) for cid in couriers]
            cur = group_cost(members, tcount)
            pool = sorted(cands, key=lambda c: (c.score, -c.willingness))
            while len(couriers) < 4:
                best_add = None
                best_cost = cur
                for c in pool:
                    if c.courier_id in used or c.courier_id in couriers:
                        continue
                    trial = members + [(c.score, c.willingness)]
                    nc = group_cost(trial, tcount)
                    if nc < best_cost - EPS:
                        best_cost = nc
                        best_add = c
                if best_add is None:
                    break
                couriers.append(best_add.courier_id)
                members.append((best_add.score, best_add.willingness))
                used.add(best_add.courier_id)
                cur = best_cost
            out.append((task_key, couriers))
        return out
