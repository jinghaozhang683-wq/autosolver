"""The AutoSolver agent: an adaptive, self-learning portfolio controller.

This is a *real* agent loop, not a fixed pipeline:

* **Autonomous exploration** - a pool of strategy "moves" (greedy, LP, exact
  CP-SAT / MILP, the pure-Python heuristic, ruin-and-recreate perturbation, and
  optional LLM reasoning). It chooses what to try next with an **upper-confidence
  bandit**, balancing exploiting the move that is paying off against exploring
  others - no human picks the algorithm.
* **Automatic evaluation & filtering** - every result is scored by the true
  objective and kept only if it beats the incumbent; the improvement is the
  move's *reward*.
* **Iterative self-adjustment (credit assignment)** - rewards update each move's
  value online. A move that keeps yielding nothing is *pivoted away from*
  (exactly the "greedy is poor at bundling -> turn elsewhere" behaviour);
  promising directions (e.g. perturbation that keeps improving) get re-tried.
* **Learning from history** - per instance-*profile* statistics are persisted to
  disk and loaded next run, so the agent starts new instances already biased
  toward strategies that worked on similar ones.
* **Anytime output within budget** - returns the best solution found so far.

`agent.log` is a human-readable trace of the agent's decisions and reasoning.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional

from .problem import Problem, group_cost, EPS
from .solution import Solution
from .profile import profile as profile_instance
from .memory import Memory
from .learning.features import solution_state
from .learning.logger import LearningLogger
from .learning.policy_model import PolicyModel
from .learning.column_model import ColumnScorer
from .learning.backup_model import BackupScorer
from .learning.llm_advisor import LlmAdvisor
from .learning.operator_model import OperatorModel
from .solvers.greedy import GreedySolver
from .solvers.lp import LpSolver
from .solvers.milp import MilpSolver
from .solvers.cpsat import CpSatSolver
from .solvers.colgen import ColGenSolver
from .solvers.heuristic import HeuristicSolver
from .solvers.llm import LlmSolver


# Warm the (slow, cold-disk) scipy import at module load so it is paid once at
# `import autosolver` rather than inside the first timed solve().
try:
    import numpy as _np_warm  # noqa: F401
    import scipy.optimize as _sp_warm  # noqa: F401
except Exception:
    pass

_SCIPY = None


def _scipy_ok():
    global _SCIPY
    if _SCIPY is None:
        try:
            import numpy, scipy.optimize  # noqa: F401
            _SCIPY = True
        except Exception:
            _SCIPY = False
    return _SCIPY


class Move:
    """One thing the agent can try."""

    __slots__ = ("name", "run", "kind", "prior", "min_budget", "params",
                 "llm_bias", "value", "tries", "stalls", "exhausted",
                 "total_time")

    def __init__(self, name: str, run: Callable, kind: str, prior: float,
                 min_budget: float, params: Optional[dict] = None):
        self.name = name
        self.run = run                # (problem, budget, incumbent) -> Solution
        self.kind = kind              # "oneshot" | "repeatable"
        self.prior = prior            # default value if no history
        self.min_budget = min_budget
        self.params = params or {}
        self.llm_bias = 0.0
        self.value = prior            # online estimate of normalised improve-rate
        self.tries = 0
        self.stalls = 0               # consecutive zero-improvement tries
        self.exhausted = False
        self.total_time = 0.0


class LogEntry:
    __slots__ = ("kind", "text")

    def __init__(self, kind, text):
        self.kind = kind
        self.text = text

    def __repr__(self):
        return self.text


class AutoSolverAgent:
    def __init__(self, use_cpsat: bool = False, use_llm: bool = False,
                 memory_path: str = "autosolver_memory.json",
                 learning_log_path: Optional[str] = "autosolver_learning.jsonl",
                 policy_model_path: Optional[str] = "autosolver_policy.json",
                 column_model_path: Optional[str] = "autosolver_column_model.json",
                 backup_model_path: Optional[str] = "autosolver_backup_model.json",
                 operator_model_path: Optional[str] = "autosolver_operator_model.json",
                 exploration: float = 0.7, seed = None,
                 verbose: bool = False, on_event=None):
        self.greedy = GreedySolver()
        self.lp = LpSolver()
        self.exact = CpSatSolver() if use_cpsat else MilpSolver()
        self.colgen = ColGenSolver()   # on-demand columns for the dense regime
        self.heuristic = HeuristicSolver()
        self.llm = LlmSolver() if use_llm else None
        self.llm_advisor = LlmAdvisor() if use_llm else None
        self.memory = Memory(memory_path)
        self.learning_logger = LearningLogger(learning_log_path)
        self.policy_model = PolicyModel(policy_model_path)
        self.column_scorer = ColumnScorer(column_model_path)
        self.backup_scorer = BackupScorer(backup_model_path)
        self.operator_model = OperatorModel(operator_model_path, seed=seed)
        if hasattr(self.exact, "column_scorer"):
            self.exact.column_scorer = self.column_scorer
            self.exact.backup_scorer = self.backup_scorer
        self.colgen.column_scorer = self.column_scorer
        self.colgen.backup_scorer = self.backup_scorer
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.on_event = on_event      # optional callback(dict) for live streaming
        self.log: List[LogEntry] = []
        self.profile: dict = {}

    # ---------------------------------------------------------------- public
    def solve(self, problem: Problem, total_budget: float = 10.0) -> Solution:
        t0 = time.time()
        instance_id = "case_%d" % int(t0 * 1000)
        self.log = []
        self.profile = profile_instance(problem)
        key = self.profile["key"]
        self._say("plan", "profile=%s  (tasks=%d couriers=%d candidates=%d)" % (
            key, self.profile["task_count"], self.profile["courier_count"],
            self.profile["candidates"]),
            event="profile", profile_key=key,
            task_count=self.profile["task_count"],
            courier_count=self.profile["courier_count"],
            candidates=self.profile["candidates"])

        self._share_columns(problem)
        # Pay the (~2-3s) scipy import up front when an exact / LP / column-
        # generation strategy will actually run; on very large instances where
        # only the heuristic runs, skip it so it doesn't burn the budget.
        if self.exact._pool is not None or len(problem.candidates) <= 20000:
            try:
                import numpy, scipy.optimize  # noqa: F401
            except Exception:
                pass

        moves = self._build_moves(problem, key)
        self._say("plan", "moves & learned priors: " + ", ".join(
            "%s=%.2f%s" % (m.name, m.value,
                           "*" if self.memory.stats(key, m.name)["tries"] else "")
            for m in moves))

        # establish a cheap baseline first so every later move is measured as a
        # genuine improvement over it (a move picked into a vacuum would
        # otherwise look value-less and the agent would mis-learn).
        incumbent: Optional[Solution] = None
        incumbent_move = ""
        baseline = 1.0
        contributions = {}   # move -> best normalised improvement contributed
        contribution_times = {}
        last_move = ""
        last_gain = 0.0
        gm = next((m for m in moves if m.name == "greedy"), None)
        if gm is not None:
            ts = time.time()
            gsol = self._safe_run(gm, problem, gm.min_budget + 0.5, None)
            gdt = max(1e-3, time.time() - ts)
            gm.tries += 1
            gm.total_time += gdt
            gm.exhausted = True
            contributions["greedy"] = 0.0
            contribution_times["greedy"] = gdt
            last_move = "greedy"
            last_gain = 0.0
            if gsol.feasible:
                incumbent = gsol
                incumbent_move = "greedy"
                baseline = max(1.0, abs(gsol.score))
                self._say("try", "%-10s -> %.4f  (baseline)" % ("greedy", gsol.score))

        self._apply_llm_advice(problem, moves, incumbent, total_budget,
                               last_move, last_gain)

        step = 0
        while time.time() - t0 < total_budget - 0.4:
            avail = [m for m in moves if not m.exhausted
                     and (total_budget - (time.time() - t0)) >= m.min_budget]
            if not avail:
                break
            remaining = total_budget - (time.time() - t0)
            select_state = solution_state(problem, incumbent, remaining, step,
                                          last_move, last_gain)
            m = self._select(avail, step, select_state)
            budget = self._slice(m, remaining, total_budget)
            state = solution_state(problem, incumbent, remaining, step,
                                   last_move, last_gain)
            step += 1

            before_score = incumbent.score if (incumbent and incumbent.feasible) else None
            ts = time.time()
            sol = self._safe_run(m, problem, budget, incumbent)
            dt = max(1e-3, time.time() - ts)
            m.total_time += dt
            m.tries += 1

            old = incumbent.score if (incumbent and incumbent.feasible) else None
            improved = sol.better_than(incumbent)
            if improved:
                incumbent = sol
                incumbent_move = m.name
            new = incumbent.score if incumbent.feasible else None

            # Time-aware reward: improvement per unit time.
            # A fast solver that finds the same improvement is valued higher
            # than a slow one — rewards efficient strategies.
            if old is not None and new is not None:
                improvement = max(0.0, old - new)
                if improvement > EPS:
                    reward = improvement / baseline / max(0.1, dt)
                else:
                    reward = 0.0
            else:
                reward = 0.0
            self._credit(m, reward)          # online value update + pivot logic
            contributions[m.name] = max(contributions.get(m.name, 0.0), reward)
            contribution_times[m.name] = contribution_times.get(m.name, 0.0) + dt
            last_move = m.name
            last_gain = reward
            self._log_learning_event(
                instance_id, key, problem, state, m, budget,
                before_score, sol, incumbent, reward, dt)

            self._say("try", "%-12s budget=%.1fs dt=%.2fs -> %s  gain=%s  value=%.3f  %s" % (
                m.name, budget, dt,
                ("%.4f" % sol.score) if sol.feasible else "infeasible",
                ("+%.4f" % reward) if reward > EPS else "0",
                m.value,
                self._move_note(m, sol)),
                event="move", move=m.name, budget=round(budget, 2),
                score=(round(sol.score, 4) if sol.feasible else None),
                gain=round(reward, 4), value=round(m.value, 4),
                feasible=sol.feasible, improved=improved,
                pivot=(m.exhausted and m.kind == "repeatable"),
                optimal=bool(getattr(sol, "optimal", False)),
                restricted_optimal=bool(getattr(sol, "restricted_optimal", False)),
                optimal_scope=getattr(sol, "optimal_scope", "none"),
                incumbent=round(incumbent.score, 4) if incumbent and incumbent.feasible else None)

            if m.kind == "oneshot":
                m.exhausted = True
            if self._provably_optimal(incumbent):
                self._say("stop", "global optimality proven by '%s' -> stop early" % m.name)
                break

        if incumbent is None or not incumbent.feasible:
            incumbent = self.greedy.solve(problem, 0.5)
            incumbent_move = "greedy"

        self._learn(key, contributions, contribution_times, incumbent_move)
        self._say("done", "best=%.4f assigned=%d/%d by '%s'  (%.2fs, %d steps)" % (
            incumbent.score, incumbent.assigned_count, problem.task_count,
            incumbent.strategy, time.time() - t0, step),
            event="done", best=round(incumbent.score, 4),
            assigned=incumbent.assigned_count, total_tasks=problem.task_count,
            winner=incumbent_move, seconds=round(time.time() - t0, 2), steps=step,
            groups=incumbent.to_output())
        return incumbent

    # ------------------------------------------------------------- internals
    def _build_moves(self, problem: Problem, key: str) -> List[Move]:
        nt = self.profile["task_count"]
        ncand = self.profile["candidates"]
        # Static MILP is reliable on a moderate pool, but low-willingness
        # instances (backup-heavy, many near-equal columns) bog it down even at
        # moderate column counts -> route those to column generation instead.
        pool_n = len(self.exact._pool) if self.exact._pool is not None else 0
        lowwill = self.profile.get("willingness_class") == "lowwill"
        exact_ok = (self.exact._pool is not None and pool_n <= 32000
                    and not (lowwill and pool_n > 12000))
        # column generation fills the gap where the static pool explodes or is
        # too hard for static MILP, but the instance is still tractable: it
        # prices columns on demand instead of enumerating them.
        colgen_ok = (not exact_ok) and _scipy_ok() and ncand <= 20000
        self._exact_present = exact_ok
        self._colgen_present = colgen_ok

        defs = []
        # (name, kind, default_prior, min_budget, runner)
        # Greedy with full backup filling (baseline, always runs first)
        defs.append(("greedy", "oneshot", 0.05, 0.05,
                     lambda p, b, inc: self.greedy.solve(p, b, inc)))
        # Greedy without backups — ultra-fast alternative for simple instances
        defs.append(("greedy_fast", "oneshot", 0.03, 0.02,
                     lambda p, b, inc: self._greedy_nobackup(p)))
        # heuristic is a bounded probe when a strong exact method runs, else the
        # main solver (gets the whole budget).
        h_prior = 0.40 if (exact_ok or colgen_ok) else 0.92
        defs.append(("heuristic", "repeatable", h_prior, 0.5,
                     lambda p, b, inc: self.heuristic.solve(p, b, inc)))
        if exact_ok:
            defs.append(("exact", "oneshot", 0.80, 1.0,
                         lambda p, b, inc: self.exact.solve(p, b, inc)))
        if colgen_ok:
            defs.append(("colgen_price24_backup4", "oneshot", 0.72, 2.0,
                         lambda p, b, inc: self._colgen_variant(p, b, inc, 24, 4)))
            defs.append(("colgen_price40_backup4", "oneshot", 0.76, 2.0,
                         lambda p, b, inc: self._colgen_variant(p, b, inc, 40, 4)))
        if ncand <= 800:
            defs.append(("lp", "oneshot", 0.25, 0.2,
                         lambda p, b, inc: self.lp.solve(p, b, inc)))
        defs.append(("perturb_add_backup", "repeatable", 0.24, 0.3,
                     lambda p, b, inc: self._perturb(p, inc, "add_backup")))
        defs.append(("perturb_drop_repair", "repeatable", 0.30, 0.7,
                     lambda p, b, inc: self._perturb(p, inc, "drop_repair")))
        if self.llm is not None and self.llm.available() and nt <= 30:
            # A reasoning LLM call has high latency (~15-17s for LongCat) and
            # costs tokens, so it is one-shot and only affordable on a generous
            # budget; within a 10s/case limit the agent correctly skips it in
            # favour of the fast solvers.
            defs.append(("llm", "oneshot", 0.35, 20.0,
                         lambda p, b, inc: self.llm.solve(p, b, inc)))

        self._exact_present = exact_ok
        moves = []
        for name, kind, prior, mb, run in defs:
            mv = Move(name, run, kind, prior, mb, params=self._move_params(name))
            mv.value = self.memory.prior_value(key, name, prior)
            moves.append(mv)
        return moves

    def _move_params(self, name: str) -> dict:
        if name == "exact":
            return {
                "topk_single": getattr(self.exact, "topk_single", 0),
                "subset_pool": getattr(self.exact, "subset_pool", 0),
                "max_backup": getattr(self.exact, "max_backup", 0),
                "max_pairs": getattr(self.exact, "max_pairs", 0),
            }
        if name.startswith("colgen_price"):
            price = 0
            try:
                price = int(name.split("_price", 1)[1].split("_", 1)[0])
            except Exception:
                pass
            return {
                "price_pool": price or getattr(self.colgen, "price_pool", 0),
                "max_backup": getattr(self.colgen, "max_backup", 0),
            }
        if name == "heuristic":
            return {"mode": 1.0}
        if name.startswith("perturb_"):
            return {"operator": name.replace("perturb_", "")}
        return {}

    def _select(self, avail: List[Move], step: int, state: dict) -> Move:
        """Contextual UCB: model prediction + history + exploration + LLM bias."""
        total = sum(m.tries for m in avail) + 1
        best = None
        best_u = -1e18
        for m in avail:
            bonus = self.exploration * math.sqrt(math.log(total + 1) / (m.tries + 1))
            action = {"move": m.name, "budget": m.min_budget, "params": m.params}
            pred = self.policy_model.predict(self.profile, state, action)
            u = pred + m.value + bonus + 0.1 * m.llm_bias
            if u > best_u:
                best_u = u
                best = m
        return best

    def _apply_llm_advice(self, problem: Problem, moves: List[Move],
                          incumbent: Optional[Solution], total_budget: float,
                          last_move: str, last_gain: float) -> None:
        if (self.llm_advisor is None or not self.llm_advisor.available()
                or total_budget < 20.0):
            return
        state = solution_state(problem, incumbent, total_budget, 0,
                               last_move, last_gain)
        history = []
        if incumbent is not None and incumbent.feasible:
            history.append({"move": last_move or incumbent.strategy,
                            "score": incumbent.score,
                            "gain": last_gain,
                            "dt": incumbent.solve_time})
        advice = self.llm_advisor.advise(
            self.profile, state, history, [m.name for m in moves],
            timeout=min(6.0, max(3.0, total_budget * 0.15)))
        bias = advice.get("action_bias") if isinstance(advice, dict) else None
        if not isinstance(bias, dict):
            return
        for m in moves:
            try:
                m.llm_bias = float(bias.get(m.name, 0.0))
            except Exception:
                m.llm_bias = 0.0
        self._say("advisor", "LongCat action bias: " + ", ".join(
            "%s=%+.2f" % (m.name, m.llm_bias)
            for m in moves if abs(m.llm_bias) > 1e-9))

    def _slice(self, m: Move, remaining: float, total: float) -> float:
        if m.name == "greedy_fast":
            return max(m.min_budget, min(remaining - 0.05, 0.1))
        if m.name.startswith("colgen"):
            return max(m.min_budget, remaining - 0.6)      # leave margin for refine
        if m.name in ("exact", "lp"):
            return max(m.min_budget, remaining - 0.1)      # exact wants it all
        if m.name == "heuristic":
            if getattr(self, "_exact_present", False) or getattr(self, "_colgen_present", False):
                # a strong exact method runs -> heuristic is a bounded probe
                return max(m.min_budget, min(remaining - 0.1, total * 0.45))
            if self.profile.get("task_count", 0) > 35:
                # very large: heuristic is the only strong move, give it all
                return max(m.min_budget, remaining - 0.1)
            # mid-size without exact: leave room for perturbation to iterate
            return max(m.min_budget, min(remaining - 0.5, total * 0.6))
        if m.name.startswith("perturb"):
            return max(m.min_budget, min(remaining - 0.05, 0.6))
        if m.name == "llm":
            return max(m.min_budget, min(remaining - 0.1, 3.0))
        return max(m.min_budget, min(remaining - 0.05, 1.0))

    def _credit(self, m: Move, reward: float) -> None:
        # online running mean of the move's normalised improvement (credit
        # assignment): moves that improve the incumbent gain value, those that
        # don't lose it.
        n = max(1, m.tries)
        m.value = (m.value * (n - 1) + reward) / n
        if reward <= 1e-6:
            m.stalls += 1
        else:
            m.stalls = 0
        # pivot: a repeatable move is abandoned after several consecutive
        # zero-improvement tries. The threshold is generous so the agent
        # keeps exploring when time budget permits — different random
        # perturbations can uncover improvements that a single attempt misses.
        if m.kind == "repeatable" and m.stalls >= 4:
            m.exhausted = True

    def _move_note(self, m: Move, sol: Solution) -> str:
        if getattr(sol, "optimal", False):
            return "[proved global optimal]"
        if getattr(sol, "restricted_optimal", False):
            return "[restricted-pool optimal]"
        if m.exhausted and m.kind == "repeatable":
            return "[stalled -> pivot away]"
        if m.kind == "oneshot":
            return "[one-shot, done]"
        return ""

    # -- greedy without backup filling (ultra-fast variant) --------------
    def _greedy_nobackup(self, problem: Problem) -> Solution:
        """Greedy construction without backup courier filling.
        Much faster than full greedy — useful as a quick alternative baseline
        on instances where backup search dominates greedy's runtime."""
        from .solvers.greedy import _saving, _ORDERS
        positive = [c for c in problem.candidates if _saving(c) > 0.001]
        best = None
        for keyfn in _ORDERS[:3]:   # top 3 orders only
            groups = []
            used_tasks, used_cours = set(), set()
            for c in sorted(positive, key=keyfn):
                if c.courier_id in used_cours:
                    continue
                if any(t in used_tasks for t in c.task_ids):
                    continue
                groups.append((c.task_key, [c.courier_id]))
                used_cours.add(c.courier_id)
                used_tasks.update(c.task_ids)
            sol = Solution(groups, strategy="greedy_fast")
            sol.evaluate(problem)
            if sol.better_than(best):
                best = sol
        return best if best is not None else Solution([], strategy="greedy_fast")

    def _colgen_variant(self, problem: Problem, budget: float,
                        incumbent: Optional[Solution],
                        price_pool: int, max_backup: int) -> Solution:
        solver = ColGenSolver(max_backup=max_backup, price_pool=price_pool)
        solver.column_scorer = self.column_scorer
        solver.backup_scorer = self.backup_scorer
        sol = solver.solve(problem, budget, incumbent)
        sol.strategy = "colgen_price%d_backup%d" % (price_pool, max_backup)
        return sol

    # -- ruin & recreate (gives the agent an 'iterate' direction) ----------
    def _perturb(self, problem: Problem, incumbent: Optional[Solution],
                 forced_operator: Optional[str] = None) -> Solution:
        if incumbent is None or not incumbent.feasible or len(incumbent.groups) < 3:
            return Solution([], strategy="perturb")
        state = {
            "current_best": incumbent.score,
            "assigned_ratio": incumbent.assigned_count / max(1, problem.task_count),
            "group_count": len(incumbent.groups),
        }
        operator = forced_operator or self.operator_model.choose(
            self.profile.get("key", ""), state, ("add_backup", "drop_repair"))
        if operator == "add_backup":
            sol = self._perturb_add_backup(problem, incumbent)
        else:
            sol = self._perturb_drop_repair(problem, incumbent)
            operator = "drop_repair"
        self._log_operator_event(problem, incumbent, sol, operator, state)
        return sol

    def _perturb_add_backup(self, problem: Problem,
                            incumbent: Optional[Solution]) -> Solution:
        groups = [(tk, list(cs)) for tk, cs in incumbent.groups]
        try:
            groups = self.greedy._add_backups(problem, groups)
        except Exception:
            return Solution([], strategy="perturb")
        sol = Solution(groups, strategy="perturb")
        sol.evaluate(problem)
        return sol

    def _perturb_drop_repair(self, problem: Problem,
                             incumbent: Optional[Solution]) -> Solution:
        groups = [(tk, list(cs)) for tk, cs in incumbent.groups]
        # drop a random ~30% of groups (weighted toward the costly ones)
        scored = []
        for tk, cs in groups:
            tcount = len(problem.bundle_tasks(tk))
            members = [(problem.info(tk, c)[0], problem.info(tk, c)[1])
                       for c in cs if problem.info(tk, c)]
            scored.append((group_cost(members, tcount) / max(1, tcount), tk, cs))
        scored.sort(reverse=True)
        n_drop = max(1, int(len(scored) * 0.3))
        drop = set()
        for i in range(n_drop):
            # roulette toward high-cost groups
            j = min(len(scored) - 1, int(self.rng.random() ** 2 * len(scored)))
            drop.add(scored[j][1])
        kept = [(tk, cs) for tk, cs in groups if tk not in drop]
        used_tasks = set(t for tk, _ in kept for t in problem.bundle_tasks(tk))
        used_cour = set(c for _, cs in kept for c in cs)
        # greedily re-cover freed tasks
        free = [c for c in problem.candidates
                if c.courier_id not in used_cour
                and not (set(c.task_ids) & used_tasks)]
        free.sort(key=lambda c: (c.cost / max(1, c.task_count)))
        for c in free:
            if c.courier_id in used_cour:
                continue
            if set(c.task_ids) & used_tasks:
                continue
            kept.append((c.task_key, [c.courier_id]))
            used_cour.add(c.courier_id)
            used_tasks.update(c.task_ids)
        sol = Solution(kept, strategy="perturb")
        sol.evaluate(problem)
        return sol

    def _log_operator_event(self, problem: Problem, before: Solution,
                            after: Solution, operator: str, state: dict) -> None:
        delta = 0.0
        accepted = False
        if before is not None and before.feasible and after is not None and after.feasible:
            delta = after.score - before.score
            accepted = after.score < before.score - EPS
        self.learning_logger.write({
            "event_type": "local_operator",
            "profile_key": self.profile.get("key", ""),
            "features": self.profile,
            "local_state_features": state,
            "operator": operator,
            "delta_score": round(delta, 6),
            "accepted": accepted,
            "score_before": before.score if before and before.feasible else None,
            "score_after": after.score if after and after.feasible else None,
        })

    def _share_columns(self, problem: Problem) -> None:
        self.exact._pool = None
        if not hasattr(self.exact, "_pool"):
            return
        if len(problem.candidates) > 12000:
            return
        try:
            from .columns import build_columns
            self.exact._pool = build_columns(
                problem,
                max_backup=getattr(self.exact, "max_backup", 4),
                topk_single=getattr(self.exact, "topk_single", 18),
                subset_pool=getattr(self.exact, "subset_pool", 13),
                max_pairs=getattr(self.exact, "max_pairs", 150),
                column_scorer=self.column_scorer,
                backup_scorer=self.backup_scorer)
        except Exception:
            self.exact._pool = None

    def _safe_run(self, m: Move, problem, budget, incumbent) -> Solution:
        try:
            sol = m.run(problem, budget, incumbent)
            if sol is None:
                sol = Solution([], strategy=m.name)
            return sol
        except Exception as e:
            self._say("warn", "move '%s' errored: %s" % (m.name, e))
            return Solution([], strategy=m.name)

    def _provably_optimal(self, incumbent) -> bool:
        return bool(incumbent and incumbent.feasible
                    and getattr(incumbent, "optimal_scope", "none") == "global")

    def _learn(self, key: str, contributions: dict, seconds: dict,
               winner_move: str) -> None:
        for name, reward in contributions.items():
            self.memory.update(key, name, reward, seconds.get(name, 0.0),
                               won=(name == winner_move))
        self.memory.save()
        self._say("learn", "updated memory[%s]; winner='%s' reinforced" % (
            key, winner_move))

    def _log_learning_event(self, instance_id: str, key: str, problem: Problem,
                            state: dict, move: Move, budget: float,
                            score_before, sol: Solution,
                            incumbent: Optional[Solution],
                            reward: float, dt: float) -> None:
        score_after = sol.score if sol.feasible else None
        best_after = incumbent.score if incumbent and incumbent.feasible else None
        improvement = 0.0
        if score_before is not None and best_after is not None:
            improvement = max(0.0, score_before - best_after)
        self.learning_logger.write({
            "instance_id": instance_id,
            "profile_key": key,
            "features": self.profile,
            "state": state,
            "action": {
                "move": move.name,
                "budget": round(budget, 4),
                "params": move.params,
            },
            "result": {
                "score_before": score_before,
                "score_after": score_after,
                "best_after": best_after,
                "improvement": round(improvement, 6),
                "dt": round(dt, 6),
                "improve_rate": round(reward, 8),
                "feasible": sol.feasible,
                "restricted_optimal": bool(getattr(sol, "restricted_optimal", False)),
                "global_optimal": bool(getattr(sol, "optimal", False)),
                "optimal_scope": getattr(sol, "optimal_scope", "none"),
                "solution_groups": sol.to_output() if sol.feasible else [],
            },
        })

    def _say(self, kind: str, text: str, **data) -> None:
        e = LogEntry(kind, "  [%-5s] %s" % (kind, text))
        self.log.append(e)
        if self.verbose:
            print(e.text)
        if self.on_event:
            payload = {"kind": kind, "text": e.text.strip()}
            payload.update(data)
            try:
                self.on_event(payload)
            except Exception:
                pass
