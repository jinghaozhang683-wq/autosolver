"""Pre-train the agent's strategy priors over a spread of synthetic instances.

    python -m autosolver.pretrain [--out PATH] [--budget 6] [--rounds 1]

Generates instances across the profile space (size x density x willingness),
runs the agent on each with a shared memory file, and prints what was learned.
The resulting memory ships with the agent so it starts new instances already
biased toward the strategies that worked on similar ones -- i.e. it arrives
"pre-trained" rather than cold.

By default writes to ``autosolver/trained_memory.json`` (the file the CLI and the
web UI read), so the learning shows up everywhere.
"""

from __future__ import annotations

import argparse
import os
import random
import time

from .problem import Problem
from .agent import AutoSolverAgent
from .memory import Memory

_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "trained_memory.json")


def _synth(seed, n_tasks, n_couriers, density, pairs, w_lo, w_hi):
    random.seed(seed)
    tasks = ["T%04d" % i for i in range(n_tasks)]
    cours = ["C%03d" % i for i in range(n_couriers)]
    lines = ["task_id_list\tstation_id\tscore\twillingness"]
    for c in cours:
        for t in tasks:
            if random.random() < density:
                lines.append("%s\t%s\t%.3f\t%.3f" % (
                    t, c, random.uniform(8, 28), random.uniform(w_lo, w_hi)))
        for _ in range(pairs):
            a, b = random.sample(tasks, 2)
            lines.append("%s,%s\t%s\t%.3f\t%.3f" % (
                a, b, c, random.uniform(18, 50), random.uniform(w_lo, w_hi)))
    return "\n".join(lines)


# a spread across the profile space
_GRID = [
    # (n_tasks, courier_mult, density, pairs, w_lo, w_hi)
    (8,  2.0, 0.6, 2, 0.6, 0.9),
    (12, 2.0, 0.5, 3, 0.6, 0.9),
    (16, 2.0, 0.5, 3, 0.5, 0.9),
    (20, 2.0, 0.5, 3, 0.5, 0.9),
    (24, 2.0, 0.45, 4, 0.5, 0.85),
    (30, 2.0, 0.4, 4, 0.5, 0.85),
    (16, 2.5, 0.5, 3, 0.1, 0.35),   # low willingness
    (24, 2.5, 0.45, 4, 0.1, 0.4),   # low willingness
    (30, 6.0, 0.9, 25, 0.4, 0.85),  # dense -> exact skipped
    (35, 6.5, 0.92, 30, 0.35, 0.85),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument("--budget", type=float, default=6.0)
    p.add_argument("--rounds", type=int, default=1,
                   help="passes over the grid (more = more confident priors)")
    p.add_argument("--fresh", action="store_true", help="start from empty memory")
    args = p.parse_args(argv)

    if args.fresh and os.path.exists(args.out):
        os.remove(args.out)

    t0 = time.time()
    n = 0
    print("pre-training -> %s  (budget %.1fs/instance)" % (args.out, args.budget))
    for r in range(args.rounds):
        for i, (nt, mult, dens, pairs, wlo, whi) in enumerate(_GRID):
            seed = 1000 * r + i
            text = _synth(seed, nt, int(nt * mult), dens, pairs, wlo, whi)
            prob = Problem.parse(text)
            agent = AutoSolverAgent(memory_path=args.out)
            sol = agent.solve(prob, args.budget)
            n += 1
            prof = agent.profile.get("key", "?")
            winner = next((a.text for a in agent.log if a.kind == "done"), "")
            print("  [%2d] %-26s tasks=%-2d cand=%-5d -> score=%.2f win=%s" % (
                n, prof, prob.task_count, len(prob.candidates),
                sol.score, sol.strategy))

    print("\ntrained on %d instances in %.1fs\n" % (n, time.time() - t0))
    # summary of learned priors
    mem = Memory(args.out)
    print("learned strategy priors per profile:")
    for key in sorted(mem.data):
        moves = mem.data[key]
        best = sorted(moves.items(), key=lambda kv: -kv[1]["improve_rate"])
        s = ", ".join("%s(win %d/%d, +%.2f)" % (
            mv, st["wins"], st["tries"], st["improve_rate"]) for mv, st in best)
        print("  %-26s %s" % (key, s))


if __name__ == "__main__":
    main()
