"""CLI runner: solve one or more case files with the agent.

    python -m autosolver.runner CASE.txt [CASE2.txt ...] [--budget 10]
                                [--cpsat] [--llm] [--quiet]

Prints, per case, the agent's strategy log and the final score / assignment,
then a summary. Exit code 0 always (so it can batch-run).
"""

from __future__ import annotations

import argparse
import sys
import time

from .problem import Problem
from .agent import AutoSolverAgent


def run_case(path, budget, use_cpsat, use_llm, verbose):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    problem = Problem.parse(text)
    agent = AutoSolverAgent(use_cpsat=use_cpsat, use_llm=use_llm, verbose=verbose)
    t0 = time.time()
    sol = agent.solve(problem, total_budget=budget)
    wall = time.time() - t0
    if verbose:
        print("case: %s" % path)
        for a in agent.log:
            print(a.text)
    print("%-40s score=%.4f assigned=%d/%d  winner=%s  %.2fs" % (
        path.split("/")[-1].split("\\")[-1], sol.score, sol.assigned_count,
        problem.task_count, sol.strategy, wall))
    return sol, problem


def main(argv=None):
    p = argparse.ArgumentParser(description="AutoSolver agent runner")
    p.add_argument("cases", nargs="+")
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--cpsat", action="store_true", help="use CP-SAT as exact engine")
    p.add_argument("--llm", action="store_true", help="enable LLM strategy")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    total = 0.0
    n = 0
    for path in args.cases:
        sol, _ = run_case(path, args.budget, args.cpsat, args.llm,
                          verbose=not args.quiet)
        if sol.feasible:
            total += sol.score
            n += 1
    if n:
        print("-" * 60)
        print("avg score over %d cases: %.4f" % (n, total / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
