"""Column-generation exact-ish solver.

The static column pool (`columns.build_columns`) enumerates only subsets of each
bundle's top couriers, so it can miss the columns an optimal solution needs
(good *backup* couriers that aren't cheapest on their own). Column generation
fixes this: it starts from a tiny pool and **prices in** exactly the columns the
LP wants.

    repeat:
        solve LP relaxation  -> dual prices  (lambda_courier >= 0, mu_task free)
        for each bundle: greedily find a courier subset whose reduced cost
            rc(S) = cost(S) - sum_{c in S} lambda_c - sum_{t in bundle} mu_t
            is negative; add those columns
    until no improving column (or time/iteration budget)
    solve the integer master (MILP) over the generated columns

Reduced cost identifies globally attractive backup structures that a static
top-k pool would never include, so on the right instances it reaches a lower
cost than the static MILP. Pure scipy/HiGHS; no CP solver needed.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import List, Optional, Tuple

from ..problem import Problem, group_cost, UNASSIGNED_PENALTY, EPS
from ..solution import Solution
from .base import Solver

_Col = Tuple[str, Tuple[str, ...], Tuple[str, ...], float, float]
# (task_key, task_ids, couriers, cost, saving)


class ColGenSolver(Solver):
    name = "colgen"

    def __init__(self, max_backup: int = 4, init_singletons: int = 8,
                 price_pool: int = 24, max_iters: int = 12):
        self.max_backup = max_backup
        self.init_singletons = init_singletons
        self.price_pool = price_pool
        self.max_iters = max_iters
        self.column_scorer = None
        self.backup_scorer = None
        self._pool = None  # for interface symmetry with other exact solvers

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        try:
            import numpy as np
            from scipy.optimize import linprog, milp, LinearConstraint, Bounds
            from scipy.sparse import coo_matrix
        except Exception:
            return self._finish(problem, [], t0)

        # per-bundle full courier candidate list (id, score, willingness)
        bundles = {}
        for tk, cands in problem.by_task_key.items():
            tids = cands[0].task_ids
            members = [(c.courier_id, c.score, c.willingness) for c in cands]
            bundles[tk] = (tids, members)

        cols = {}        # (task_key, couriers) -> _Col

        def add(tk, tids, combo):
            cids = tuple(sorted(ci[0] for ci in combo))
            if (tk, cids) in cols:
                return False
            cost = group_cost([(ci[1], ci[2]) for ci in combo], len(tids))
            cols[(tk, cids)] = (tk, tids, cids, cost,
                                UNASSIGNED_PENALTY * len(tids) - cost)
            return True

        # init: cheapest singletons per bundle
        for tk, (tids, members) in bundles.items():
            base = UNASSIGNED_PENALTY * len(tids)
            best = sorted(members, key=lambda m: m[1] * m[2] + (1 - m[2]) * base)
            for ci in best[:self.init_singletons]:
                add(tk, tids, (ci,))

        def matrix(collist):
            tasks = sorted(set(t for c in collist for t in c[1]))
            cours = sorted(set(k for c in collist for k in c[2]))
            ti = {t: i for i, t in enumerate(tasks)}
            ki = {k: len(tasks) + i for i, k in enumerate(cours)}
            rows, jc = [], []
            for j, c in enumerate(collist):
                for t in c[1]:
                    rows.append(ti[t]); jc.append(j)
                for k in c[2]:
                    rows.append(ki[k]); jc.append(j)
            A = coo_matrix((np.ones(len(rows)), (rows, jc)),
                           shape=(len(tasks) + len(cours), len(collist))).tocsr()
            return A, tasks, cours

        lp_deadline = t0 + time_budget * 0.6
        last_lp_fun = None      # objective of the last solved LP relaxation
        # ---- column generation loop -------------------------------------
        for _ in range(self.max_iters):
            if time.time() > lp_deadline:
                break
            collist = list(cols.values())
            A, tasks, cours = matrix(collist)
            cobj = np.array([-c[4] for c in collist])
            res = linprog(cobj, A_ub=A, b_ub=np.ones(len(tasks) + len(cours)),
                          bounds=(0, 1), method="highs")
            if res.x is None:
                break
            last_lp_fun = res.fun
            y = res.ineqlin.marginals
            nt = len(tasks)
            ytask = {t: y[i] for i, t in enumerate(tasks)}
            ycour = {k: y[nt + i] for i, k in enumerate(cours)}

            added = 0
            for tk, (tids, members) in bundles.items():
                yt = sum(ytask.get(t, 0.0) for t in tids)
                # rank couriers by (single cost - dual) and greedily build
                ranked = sorted(
                    members,
                    key=lambda m: (
                        group_cost([(m[1], m[2])], len(tids))
                        - ycour.get(m[0], 0.0)
                        - self._learned_price_bias(problem, tk, m)
                    ))[:self.price_pool]
                chosen, ids = [], set()
                cur = UNASSIGNED_PENALTY * len(tids)
                for _step in range(min(self.max_backup, len(ranked))):
                    best, best_adj = None, cur - sum(ycour.get(c[0], 0.0) for c in chosen)
                    for ci in ranked:
                        if ci[0] in ids:
                            continue
                        nc = group_cost([(c[1], c[2]) for c in chosen]
                                        + [(ci[1], ci[2])], len(tids))
                        adj = nc - (sum(ycour.get(c[0], 0.0) for c in chosen)
                                    + ycour.get(ci[0], 0.0))
                        if best is None or adj < best_adj - 1e-9:
                            best, best_adj, best_nc = ci, adj, nc
                    if best is None:
                        break
                    chosen.append(best); ids.add(best[0]); cur = best_nc
                    rc = -(UNASSIGNED_PENALTY * len(tids) - cur) \
                        - sum(ycour.get(c[0], 0.0) for c in chosen) - yt
                    if rc < -1e-7:
                        if add(tk, tids, tuple(chosen)):
                            added += 1
            if added == 0:
                break

        # ---- integer master over the generated columns ------------------
        collist = list(cols.values())
        if not collist:
            return self._finish(problem, [], t0)
        A, tasks, cours = matrix(collist)
        cobj = np.array([-c[4] for c in collist])
        remaining = max(0.2, time_budget - (time.time() - t0))
        try:
            res = milp(c=cobj, constraints=LinearConstraint(A, -np.inf, 1),
                       integrality=np.ones(len(collist)), bounds=Bounds(0, 1),
                       options={"time_limit": remaining, "mip_rel_gap": 0.0})
        except Exception:
            return self._finish(problem, [], t0)
        groups = []
        if res.x is not None:
            for j, c in enumerate(collist):
                if res.x[j] > 0.5:
                    groups.append((c[0], list(c[2])))
        sol = self._finish(problem, groups, t0)
        # CG reaches the LP optimum but the integer master over generated
        # columns is not a global integer certificate -> don't claim optimal.
        sol.restricted_optimal = (getattr(res, "status", 1) == 0)
        sol.optimal = False
        sol.optimal_scope = "restricted" if sol.restricted_optimal else "none"
        # LP-relaxation reference value over the generated columns:
        #   score = C_max - total_saving, and the LP maximises saving, so
        #   C_max + lp_fun (= C_max - saving_LP) is a lower-bound *estimate*
        #   on the score. Because the pricing subproblem is solved greedily
        #   (capped by price_pool / max_backup), this is NOT a certified global
        #   bound — it is exposed only for gap reporting and never sets
        #   optimal_scope.
        if last_lp_fun is not None:
            sol.bound = UNASSIGNED_PENALTY * problem.task_count + float(last_lp_fun)
        return sol

    def _learned_price_bias(self, problem: Problem, task_key: str, member) -> float:
        if self.column_scorer is None:
            return 0.0
        try:
            from ..problem import Candidate
            tids = problem.bundle_tasks(task_key)
            cand = Candidate(tids, task_key, member[0], member[1], member[2])
            return 0.01 * self.column_scorer.predict(problem, task_key, [cand])
        except Exception:
            return 0.0
