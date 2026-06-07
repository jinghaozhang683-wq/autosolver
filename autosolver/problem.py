"""Problem model for the dispatch-assignment task.

Input format (tab separated, optional header line starting with 'task_id_list'):

    task_id_list \t station_id \t score \t willingness

`task_id_list` is a single task id ("T0001") or a comma-joined bundle
("T0001,T0002") meaning those tasks are delivered together by one courier.
`station_id` is the courier id. `score` is the predicted cost contribution and
`willingness` in [0,1] is the probability the courier accepts the order.

A *candidate* is one row: a (bundle, courier) pair with its score/willingness.
"""

from __future__ import annotations

from math import prod
from typing import Dict, List, Sequence, Tuple

UNASSIGNED_PENALTY = 100.0
EPS = 1e-9


class Candidate:
    __slots__ = ("task_ids", "task_key", "courier_id", "score",
                 "willingness", "task_count", "cost")

    def __init__(self, task_ids: Tuple[str, ...], task_key: str,
                 courier_id: str, score: float, willingness: float):
        self.task_ids = task_ids
        self.task_key = task_key
        self.courier_id = courier_id
        self.score = score
        self.willingness = willingness
        self.task_count = len(task_ids)
        # single-courier expected cost for this assignment
        base = UNASSIGNED_PENALTY * self.task_count
        self.cost = willingness * score + (1.0 - willingness) * base

    def __repr__(self) -> str:
        return "Candidate(%s, %s, s=%.3f, w=%.3f)" % (
            self.task_key, self.courier_id, self.score, self.willingness)


def group_cost(members: Sequence[Tuple[float, float]], task_count: int) -> float:
    """Expected cost of assigning a bundle (task_count tasks) to a *group* of
    couriers, each described by (score, willingness).

    The first courier to accept handles the order; failure (nobody accepts)
    incurs the unassigned penalty.

        fail_prob      = prod(1 - w_i)
        p_complete     = 1 - fail_prob
        expected_score = sum(w_i * s_i) / sum(w_i)
        cost           = p_complete * expected_score + fail_prob * 100 * task_count
    """
    if not members:
        return UNASSIGNED_PENALTY * task_count
    fail_prob = 1.0
    weighted = 0.0
    wsum = 0.0
    for score, w in members:
        fail_prob *= (1.0 - w)
        weighted += w * score
        wsum += w
    if wsum <= EPS:
        return UNASSIGNED_PENALTY * task_count
    p_complete = 1.0 - fail_prob
    expected = weighted / wsum
    return p_complete * expected + fail_prob * UNASSIGNED_PENALTY * task_count


class Problem:
    """Parsed problem instance with fast lookup structures."""

    def __init__(self, candidates: List[Candidate]):
        self.candidates = candidates
        tasks = set()
        couriers = set()
        by_task_key: Dict[str, List[Candidate]] = {}
        info: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
        for c in candidates:
            couriers.add(c.courier_id)
            for t in c.task_ids:
                tasks.add(t)
            by_task_key.setdefault(c.task_key, []).append(c)
            info[(c.task_key, c.courier_id)] = (c.score, c.willingness, c.task_count)
        self.tasks = sorted(tasks)
        self.couriers = sorted(couriers)
        self.by_task_key = by_task_key
        self._info = info
        self.task_count = len(self.tasks)
        self.courier_count = len(self.couriers)

    # -- lookups -----------------------------------------------------------
    def info(self, task_key: str, courier_id: str):
        """(score, willingness, task_count) for a (bundle, courier) or None."""
        return self._info.get((task_key, courier_id))

    def bundle_tasks(self, task_key: str) -> Tuple[str, ...]:
        return tuple(p.strip() for p in task_key.split(",") if p.strip())

    # -- parsing -----------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Problem":
        raw_best: Dict[Tuple[str, str], Candidate] = {}
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return cls([])
        start = 1 if lines[0].startswith("task_id_list") else 0
        for line in lines[start:]:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            task_key, courier_id, score_s, will_s = parts[:4]
            task_key = (task_key or "").strip()
            courier_id = (courier_id or "").strip()
            if not task_key or not courier_id:
                continue
            try:
                score = float(score_s)
                willingness = float(will_s)
            except (TypeError, ValueError):
                continue
            task_ids = tuple(p.strip() for p in task_key.split(",") if p.strip())
            if not task_ids:
                continue
            willingness = max(0.0, min(1.0, willingness))
            cand = Candidate(task_ids, task_key, courier_id, score, willingness)
            key = (task_key, courier_id)
            prev = raw_best.get(key)
            # keep the cheaper duplicate (same bundle+courier)
            if prev is None or cand.cost < prev.cost - EPS:
                raw_best[key] = cand
        return cls(list(raw_best.values()))
