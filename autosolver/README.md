# AutoSolver — multi-strategy dispatch-assignment agent

A complete **AutoSolver Agent** for the courier dispatch-assignment problem:
assign every delivery task (or bundle) to one or more couriers so that the
number of accepted orders is maximised and the total expected score (cost) is
minimised, within **10 s / test case**.

Unlike the single-file judge engine (`../solver.py`, written for an environment
with **no third-party libraries**), this package is the full system and is free
to use `ortools`, `scipy`, and an LLM API.

## What the agent does (matches the task deliverables)

It is a genuine **adaptive, self-learning agent loop**, not a fixed pipeline:

1. **Autonomous strategy exploration** — a pool of strategy *moves* (greedy, LP,
   exact CP-SAT / MILP, the pure-Python heuristic, ruin-and-recreate
   perturbation, optional LLM reasoning). The agent decides what to try next with
   an **upper-confidence bandit** — exploiting the move that is paying off while
   still exploring others. No human picks the algorithm.
2. **Automatic evaluation & filtering** — every result is scored by the true
   objective and validated; it becomes the incumbent only if it improves on it.
   The size of the improvement is the move's **reward**.
3. **Iterative self-adjustment (credit assignment)** — rewards update each move's
   value online. A move that keeps returning nothing is **pivoted away from**
   (exactly the "greedy is poor at bundling → turn elsewhere" behaviour);
   promising directions get re-tried. Only a global optimality certificate stops
   the loop; a restricted column-pool certificate is shown separately.
4. **Learning from history** — per instance-*profile* statistics
   (`size/density/willingness` bucket) are **persisted to disk** and reloaded next
   run, so the agent starts a new instance already biased toward strategies that
   worked on similar ones.
5. **Anytime output within budget** — returns the best solution found; wall time
   stays well under 10 s on every test instance.

Every decision is recorded in `agent.log` as a readable trace, e.g.:

```
[plan ] profile=large/dense/midwill (tasks=35 couriers=220 candidates=13641)
[plan ] moves & learned priors: greedy=0.05, heuristic=0.40, perturb=0.30
[try  ] greedy     -> 326.9485  (baseline)
[try  ] heuristic  budget=6.0s -> 387.65  gain=0  value=0.000
[try  ] perturb    budget=0.6s -> 429.68  gain=0  value=0.000
[try  ] heuristic  budget=3.4s -> 513.19  gain=0  [stalled -> pivot away]
[learn] updated memory[large/dense/midwill]; winner='greedy' reinforced
[done ] best=326.9485 by 'greedy' (8.6s, 4 steps)
```

## Objective

For a group of couriers assigned to a bundle of `n` tasks, each courier `i` with
predicted score `s_i` and acceptance probability (willingness) `w_i`:

```
fail_prob      = prod(1 - w_i)
p_complete     = 1 - fail_prob
expected_score = sum(w_i * s_i) / sum(w_i)
group_cost     = p_complete * expected_score + fail_prob * 100 * n
```

An uncovered task costs `100`. Total cost = sum of group costs + uncovered
penalty. Each task is covered at most once; each courier is used at most once.

## Architecture

```
autosolver/
  problem.py     Problem model: parse input, Candidate, group_cost
  solution.py    Solution + true-objective evaluation / feasibility check
  columns.py     compact high-quality column (set-partition variable) pool
  solvers/
    base.py      common Solver interface
    greedy.py    multi-order greedy + backup filling  (sub-second baseline)
    lp.py        LP relaxation (HiGHS) + rounding      (fast, small instances)
    milp.py      exact set partition via scipy.optimize.milp / HiGHS  (default)
    cpsat.py     exact set partition via OR-Tools CP-SAT  (--cpsat)
    colgen.py    column generation: prices columns on demand (dense regime)
    heuristic.py pure-Python heuristic wrapper with an internal safe fallback
    llm.py       LLM direct reasoning via the Anthropic API  (--llm)
  profile.py     instance features -> discrete profile key (for learning)
  memory.py      persistent cross-run learning store (JSON)
  learning/      JSONL state-action-result traces for policy training
    policy_model.py / train.py        contextual bandit value model
    column_model.py / backup_model.py learned column-pool guidance hooks
    llm_advisor.py                    LongCat meta-controller bias
    operator_model.py / train_operator.py local-search operator selector
  agent.py       adaptive bandit controller: explore, credit-assign, pivot, learn
  runner.py      CLI
```

### The agent loop (agent.py)

* Each **move** has an online *value* = mean normalised improvement it has
  produced. It is seeded from `memory.py` (history on similar instances) or a
  default prior.
* A cheap **greedy baseline** is taken first; every later move's reward is its
  improvement over the incumbent, normalised by the baseline score.
* Move selection is **UCB**: `value + c·sqrt(ln N / (n+1))` — exploit good moves,
  explore rarely-tried ones.
* If `autosolver_policy.json` exists, move selection becomes contextual:
  `policy_prediction + historical_value + exploration + 0.1 * llm_bias`.
  Actions are named with strategy parameters where useful, e.g.
  `colgen_price40_backup4` and `perturb_add_backup`.
* After each move the value is updated; a repeatable move that yields nothing
  twice in a row is **abandoned (pivot)**. One-shot moves (exact, LP) run once.
* When a solver **proves global** optimality the loop stops early. Restricted
  column-pool certificates are recorded but do not stop the anytime loop.
* On finish, the per-move rewards and the winner are written back to
  `autosolver_memory.json` — the learning that carries to the next run.
* Each move attempt is also appended to `autosolver_learning.jsonl` as a
  state-action-result sample for future contextual-bandit training.
* Local-search operators append `local_operator` samples; train them with
  `python -m autosolver.learning.train_operator`.

### How exact solving handles optimality

The non-linear objective is linearised by **column enumeration**: each
`(bundle, courier-subset)` becomes a variable with a precomputed constant cost.
The solver then picks columns to cover each task ≤ 1 and use each courier ≤ 1,
maximising total saving — a set-partition ILP that CP-SAT / HiGHS solve over
the generated column pool. Because `topk_single`, `subset_pool`, `max_backup`,
`max_pairs`, and `column_cap` can prune that pool, a solver status of "optimal"
usually means **restricted-pool optimal**, not globally optimal for the original
problem. `Solution.optimal_scope` distinguishes `"global"`, `"restricted"`,
and `"none"`.

When the static pool would explode or become too hard for the integer solver
(dense / low-willingness instances), the agent switches to **column generation**
(`colgen.py`): it starts from a tiny pool, solves the LP relaxation, and prices
in only the columns the duals say are attractive — including good *backup*
couriers a static top-k pool would miss — then solves the integer master over
the generated columns. This rescues exactly the regime where static MILP returns
nothing (e.g. it turns a failed 2200 into ~785 on a 24k-column low-willingness
instance). The very largest dense instances fall back to the heuristic.

### Pre-training the strategy priors

`python -m autosolver.pretrain` runs the agent over a spread of synthetic
instances and writes per-profile strategy statistics to `trained_memory.json`,
which the CLI and web UI load by default — so the agent ships **pre-trained**:
on a new instance it already knows, e.g., that small/sparse instances are won by
the exact solver while dense ones favour greedy / column generation.

## Install

```
pip install ortools scipy numpy anthropic
```

`scipy` (MILP/LP) is required for the exact/LP strategies; `ortools` for the
`--cpsat` engine; `anthropic` only for `--llm`. The heuristic strategy is pure
Python and always available.

## Usage

**Web UI** (recommended for demos) — watch the agent think live, see the
solution, the per-strategy values, and the learning memory build up across runs:

```
pip install flask
python -m autosolver.web.server          # -> http://127.0.0.1:5000
```

Self-contained CLI demo (generates its own data, shows the full agent behaviour:
exact + early-stop, exploration + pivoting, and cross-run learning):

```
python -m autosolver.demo [--llm]
```

CLI on real case files:

```
python -m autosolver.runner CASE.txt [CASE2.txt ...] [--budget 10]
                            [--cpsat] [--llm] [--quiet]
```

Python:

```python
from autosolver import Problem
from autosolver.agent import AutoSolverAgent

problem = Problem.parse(open("case.txt").read())
agent = AutoSolverAgent(use_cpsat=False, use_llm=False, verbose=True)
solution = agent.solve(problem, total_budget=10.0)
print(solution.score, solution.to_output())
for attempt in agent.log:        # the autonomous exploration trace
    print(attempt)
```

### LLM strategy

Set an API key to enable it:

```
setx ANTHROPIC_API_KEY "sk-ant-..."      # Windows
export ANTHROPIC_API_KEY=sk-ant-...      # bash
python -m autosolver.runner case.txt --llm
```

Without a key the LLM strategy reports itself unavailable and the agent simply
skips it.

## Results (local)

| instance        | tasks | candidates | agent score | winner    | wall |
|-----------------|------:|-----------:|------------:|-----------|-----:|
| synth tiny      |     6 |         57 |      109.27 | milp opt  | ~4 s |
| synth small     |    12 |        181 |      214.65 | milp opt  | ~5 s |
| synth medium    |    20 |        497 |      291.35 | milp opt  | ~4 s |
| large_seed301   |    40 |     33 780 |      649.94 | heuristic | ~2 s |

On small/medium instances the exact solver proves the optimum and beats the
heuristic; on the largest dense instance exact is skipped and the heuristic
(equivalent to the tuned judge engine) is used.
