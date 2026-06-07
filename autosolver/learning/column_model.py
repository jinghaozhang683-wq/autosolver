"""Learned/heuristic column scorer for MILP and column generation."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, Mapping, Sequence

from ..problem import Candidate, Problem, UNASSIGNED_PENALTY, group_cost
from ..columns import Column


def _stats(values: Sequence[float]):
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, min(values), math.sqrt(var)


def column_features(problem: Problem, task_key: str,
                    candidates: Sequence[Candidate]) -> Dict[str, float]:
    scores = [c.score for c in candidates]
    wills = [c.willingness for c in candidates]
    tcount = len(problem.bundle_tasks(task_key))
    cost = group_cost([(c.score, c.willingness) for c in candidates], tcount)
    fail_prob = 1.0
    for c in candidates:
        fail_prob *= (1.0 - c.willingness)
    score_mean, score_min, score_std = _stats(scores)
    will_mean, will_min, will_std = _stats(wills)
    return {
        "bundle_size": float(tcount),
        "num_couriers": float(len(candidates)),
        "group_cost": cost,
        "saving": UNASSIGNED_PENALTY * tcount - cost,
        "avg_willingness": will_mean,
        "min_willingness": will_min,
        "willingness_std": will_std,
        "score_mean": score_mean,
        "score_min": score_min,
        "score_std": score_std,
        "fail_prob": fail_prob,
        "p_complete": 1.0 - fail_prob,
        "task_candidate_count": float(len(problem.by_task_key.get(task_key, []))),
    }


class ColumnScorer:
    def __init__(self, path: str | None = None):
        self.path = path
        self.weights: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.weights = {str(k): float(v)
                            for k, v in (data.get("weights") or {}).items()}
        except Exception:
            self.weights = {}

    def predict_features(self, feats: Mapping[str, float]) -> float:
        if self.weights:
            return sum(self.weights.get(k, 0.0) * float(v)
                       for k, v in feats.items())
        # Neutral learned fallback: prefer high saving and reliable completion.
        return feats.get("saving", 0.0) + 10.0 * feats.get("p_complete", 0.0)

    def predict(self, problem: Problem, task_key: str,
                candidates: Sequence[Candidate]) -> float:
        return self.predict_features(column_features(problem, task_key, candidates))

    def rank_key(self, col: Column) -> float:
        return col.saving
