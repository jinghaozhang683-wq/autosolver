"""Solution representation and the objective evaluation."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .problem import Problem, UNASSIGNED_PENALTY, EPS, group_cost


class Solution:
    """An assignment: list of (task_key, [courier_ids]) groups.

    Each group means the bundle `task_key` is offered to those couriers
    (primary + backups). `score` is the objective value (lower is better),
    computed and validated against a Problem via `evaluate`.
    """

    __slots__ = ("groups", "score", "assigned_count", "feasible", "strategy",
                 "solve_time", "optimal", "restricted_optimal",
                 "optimal_scope", "bound", "degraded")

    def __init__(self, groups: List[Tuple[str, List[str]]],
                 strategy: str = ""):
        # normalise: drop empty groups, copy courier lists
        self.groups = [(tk, list(cs)) for tk, cs in groups if cs]
        self.score: float = float("inf")
        self.assigned_count: int = 0
        self.feasible: bool = False
        self.strategy = strategy
        self.solve_time: float = 0.0
        # `optimal` is reserved for a certificate over the original problem.
        # Restricted column-pool certificates use `restricted_optimal` instead.
        self.optimal: bool = False
        self.restricted_optimal: bool = False
        self.optimal_scope: str = "none"
        self.bound: float = float("nan")
        # True when a solver silently fell back to a weaker engine (e.g. the
        # heuristic's solver.py is missing and it degraded to greedy).
        self.degraded: bool = False

    def evaluate(self, problem: Problem) -> "Solution":
        """Compute objective score + feasibility against the problem.

        Infeasible (duplicate task, duplicate courier, unknown pair) solutions
        get score = +inf so the agent discards them.
        """
        covered = set()
        used_couriers = set()
        total = 0.0
        feasible = True
        for task_key, couriers in self.groups:
            members = []
            for cid in couriers:
                row = problem.info(task_key, cid)
                if row is None:
                    feasible = False
                    break
                if cid in used_couriers:
                    feasible = False
                    break
                used_couriers.add(cid)
                members.append((row[0], row[1]))
            if not feasible:
                break
            task_ids = problem.bundle_tasks(task_key)
            tcount = len(task_ids)
            for t in task_ids:
                if t in covered:
                    feasible = False
                    break
                covered.add(t)
            if not feasible:
                break
            total += group_cost(members, tcount)

        if not feasible:
            self.feasible = False
            self.score = float("inf")
            self.assigned_count = 0
            return self

        total += UNASSIGNED_PENALTY * (problem.task_count - len(covered))
        self.feasible = True
        self.score = total
        self.assigned_count = len(covered)
        return self

    def better_than(self, other: Optional["Solution"]) -> bool:
        """Lower score wins; ties broken by more tasks assigned."""
        if other is None or not other.feasible:
            return self.feasible
        if not self.feasible:
            return False
        if self.score < other.score - EPS:
            return True
        if abs(self.score - other.score) <= EPS:
            return self.assigned_count > other.assigned_count
        return False

    def to_output(self) -> List[Tuple[str, List[str]]]:
        return [(tk, list(cs)) for tk, cs in self.groups]

    def __repr__(self) -> str:
        return "Solution(strategy=%s, score=%.4f, assigned=%d, feasible=%s)" % (
            self.strategy, self.score, self.assigned_count, self.feasible)
