"""AutoSolver: a multi-strategy agent for dispatch-assignment optimisation."""

from .problem import Problem, Candidate, group_cost
from .solution import Solution

__all__ = ["Problem", "Candidate", "group_cost", "Solution"]
