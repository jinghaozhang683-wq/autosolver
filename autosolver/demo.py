"""Self-contained demonstration of the AutoSolver agent.

    python -m autosolver.demo [--llm]

Shows, with full decision traces:
  1. exact solving + early stop on an easy instance,
  2. autonomous exploration + pivoting when no exact solver fits,
  3. cross-run learning (the same instance solved twice; run 2 starts from the
     priors learned in run 1).

Generates its own small synthetic instances, so it needs no external data.
Uses a throwaway memory file so it does not disturb a real one.
"""

from __future__ import annotations

import os
import random
import tempfile

from .problem import Problem
from .agent import AutoSolverAgent


def _synth(seed, n_tasks, n_couriers, density=0.5, pairs=3):
    random.seed(seed)
    tasks = ["T%04d" % i for i in range(n_tasks)]
    cours = ["C%03d" % i for i in range(n_couriers)]
    lines = ["task_id_list\tstation_id\tscore\twillingness"]
    for c in cours:
        for t in tasks:
            if random.random() < density:
                lines.append("%s\t%s\t%.3f\t%.3f" % (
                    t, c, random.uniform(8, 28), random.uniform(0.4, 0.9)))
        for _ in range(pairs):
            a, b = random.sample(tasks, 2)
            lines.append("%s,%s\t%s\t%.3f\t%.3f" % (
                a, b, c, random.uniform(18, 50), random.uniform(0.4, 0.85)))
    return "\n".join(lines)


def _banner(text):
    print("\n" + "=" * 68 + "\n  " + text + "\n" + "=" * 68)


def main(use_llm=False):
    mem = os.path.join(tempfile.gettempdir(), "autosolver_demo_memory.json")
    if os.path.exists(mem):
        os.remove(mem)

    _banner("1) Easy instance: exact solving proves the optimum, agent stops early")
    prob = Problem.parse(_synth(1, 16, 30))
    AutoSolverAgent(memory_path=mem, use_llm=use_llm, verbose=True).solve(prob, 10.0)

    _banner("2) No exact solver fits: autonomous exploration + pivoting")
    prob = Problem.parse(_synth(7, 35, 220, density=0.92, pairs=30))
    AutoSolverAgent(memory_path=mem, use_llm=use_llm, verbose=True).solve(prob, 10.0)

    _banner("3) Cross-run learning: same instance twice; run 2 uses learned priors")
    prob = Problem.parse(_synth(3, 18, 36))
    print("\n--- run 1 (cold) ---")
    AutoSolverAgent(memory_path=mem, verbose=True).solve(prob, 10.0)
    print("\n--- run 2 (priors marked '*' are learned from run 1) ---")
    AutoSolverAgent(memory_path=mem, verbose=True).solve(prob, 10.0)

    os.remove(mem) if os.path.exists(mem) else None
    print("\n(demo complete)")


if __name__ == "__main__":
    import sys
    main(use_llm="--llm" in sys.argv)
