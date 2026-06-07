"""Column (set-partition variable) enumeration.

A *column* is a (bundle, courier-subset) pair with a precomputed constant cost
(the non-linear group cost). The exact solvers (CP-SAT / MILP / LP) select a
set of columns so each task is covered at most once and each courier used at
most once, minimising total cost + penalty for uncovered tasks.

Enumerating every subset is exponential and most are useless, so we generate a
*compact high-quality* pool:

* singleton columns from the top `topk_single` couriers of a bundle;
* multi-courier columns (size 2..max_backup) only from the bundle's best
  `subset_pool` couriers (the ones an optimal group would actually use);
* bundles of >=2 tasks ("pairs") are pruned to the `max_pairs` most promising
  ones (highest best-column saving), since the input usually contains hundreds
  of mostly-worthless pair candidates.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, NamedTuple, Tuple

from .problem import Problem, group_cost, UNASSIGNED_PENALTY, EPS


class Column(NamedTuple):
    task_key: str
    task_ids: Tuple[str, ...]
    couriers: Tuple[str, ...]
    cost: float           # expected group cost
    saving: float         # UNASSIGNED_PENALTY*task_count - cost  (>0 = worth it)


def to_matrix(cols):
    """Build the set-partition constraint matrix (task rows + courier rows) as a
    fast CSR sparse matrix using COO triplets. Returns (A_csr, tasks, couriers).
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    tasks = sorted(set(t for c in cols for t in c.task_ids))
    cours = sorted(set(k for c in cols for k in c.couriers))
    ti = {t: i for i, t in enumerate(tasks)}
    ki = {k: len(tasks) + i for i, k in enumerate(cours)}
    rows = []
    cols_idx = []
    for j, c in enumerate(cols):
        for t in c.task_ids:
            rows.append(ti[t]); cols_idx.append(j)
        for k in c.couriers:
            rows.append(ki[k]); cols_idx.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    A = coo_matrix((data, (rows, cols_idx)),
                   shape=(len(tasks) + len(cours), len(cols))).tocsr()
    return A, tasks, cours


def _bundle_columns(problem, cands, tcount, topk_single, subset_pool, max_backup,
                    keep_nonpositive, column_scorer=None, backup_scorer=None):
    task_key = cands[0].task_key
    task_ids = cands[0].task_ids
    base = UNASSIGNED_PENALTY * tcount
    if backup_scorer is not None:
        by_cost = sorted(
            cands,
            key=lambda c: (-backup_scorer.predict(problem, task_key, [], c),
                           c.cost))
    else:
        by_cost = sorted(cands, key=lambda c: c.cost)
    cols = []

    # singleton columns (broad)
    for c in by_cost[:topk_single]:
        cost = group_cost([(c.score, c.willingness)], tcount)
        saving = base - cost
        if saving > EPS or keep_nonpositive:
            cols.append(Column(task_key, task_ids, (c.courier_id,), cost, saving))

    # multi-courier columns (narrow: only from the very best couriers)
    pool = by_cost[:subset_pool]
    if backup_scorer is not None and pool:
        seed = pool[0]
        rest = sorted(pool[1:],
                      key=lambda c: (-backup_scorer.predict(
                          problem, task_key, [seed], c), c.cost))
        pool = [seed] + rest
    for size in range(2, min(max_backup, len(pool)) + 1):
        for combo in combinations(pool, size):
            members = [(c.score, c.willingness) for c in combo]
            cost = group_cost(members, tcount)
            saving = base - cost
            if saving > EPS or keep_nonpositive:
                cols.append(Column(task_key, task_ids,
                                   tuple(c.courier_id for c in combo),
                                   cost, saving))
    if column_scorer is not None:
        cand_by_id = {c.courier_id: c for c in cands}
        cols.sort(key=lambda col: column_scorer.predict(
            problem, col.task_key, [cand_by_id[cid] for cid in col.couriers]),
            reverse=True)
    return cols


def build_columns(problem: Problem,
                  max_backup: int = 4,
                  topk_single: int = 12,
                  subset_pool: int = 7,
                  max_pairs: int = 120,
                  keep_nonpositive: bool = False,
                  column_scorer=None,
                  backup_scorer=None) -> List[Column]:
    """Enumerate a compact, high-quality column pool for the set partition."""
    singles = []
    pairs = []
    for task_key, cands in problem.by_task_key.items():
        if cands[0].task_count >= 2:
            pairs.append((task_key, cands))
        else:
            singles.append((task_key, cands))

    columns: List[Column] = []
    for task_key, cands in singles:
        columns.extend(_bundle_columns(
            problem, cands, 1, topk_single, subset_pool, max_backup,
            keep_nonpositive, column_scorer, backup_scorer))

    # rank pair bundles by their best achievable saving, keep top max_pairs
    scored_pairs = []
    for task_key, cands in pairs:
        if column_scorer is not None:
            best = max(column_scorer.predict(problem, task_key, [c])
                       for c in cands)
        else:
            best = max(c.task_count * UNASSIGNED_PENALTY -
                       group_cost([(c.score, c.willingness)], c.task_count)
                       for c in cands)
        scored_pairs.append((best, task_key, cands))
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    for _, task_key, cands in scored_pairs[:max_pairs]:
        columns.extend(_bundle_columns(
            problem, cands, cands[0].task_count, topk_single, subset_pool,
            max_backup, keep_nonpositive, column_scorer, backup_scorer))

    return columns
