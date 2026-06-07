"""Backup-courier scorer used when constructing candidate column pools."""

from __future__ import annotations

import json
import os
from typing import Dict, Mapping, Sequence

from ..problem import Candidate, Problem, group_cost


def backup_features(problem: Problem, task_key: str, selected: Sequence[Candidate],
                    candidate: Candidate) -> Dict[str, float]:
    tcount = len(problem.bundle_tasks(task_key))
    before = group_cost([(c.score, c.willingness) for c in selected], tcount)
    after = group_cost([(c.score, c.willingness) for c in selected]
                       + [(candidate.score, candidate.willingness)], tcount)
    fail_before = 1.0
    for c in selected:
        fail_before *= (1.0 - c.willingness)
    fail_after = fail_before * (1.0 - candidate.willingness)
    return {
        "delta_group_cost": before - after,
        "delta_fail_prob": fail_before - fail_after,
        "score": candidate.score,
        "willingness": candidate.willingness,
        "single_cost": candidate.cost,
        "selected_size": float(len(selected)),
        "task_candidate_count": float(len(problem.by_task_key.get(task_key, []))),
    }


class BackupScorer:
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
        return (
            2.0 * feats.get("delta_group_cost", 0.0)
            + 20.0 * feats.get("delta_fail_prob", 0.0)
            + feats.get("willingness", 0.0)
            - 0.01 * feats.get("score", 0.0)
        )

    def predict(self, problem: Problem, task_key: str,
                selected: Sequence[Candidate], candidate: Candidate) -> float:
        return self.predict_features(
            backup_features(problem, task_key, selected, candidate))
