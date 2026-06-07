"""Contextual bandit policy-value model.

The model is intentionally small: a JSON-serialised linear regressor trained
from `autosolver_learning.jsonl`. If no trained file exists, predictions are
neutral and the agent falls back to historical memory + UCB exploration.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Mapping, Optional


_NUMERIC_FEATURES = (
    "task_count", "courier_count", "candidates", "cands_per_task",
    "candidate_count", "density", "pair_ratio", "avg_willingness",
    "willingness_mean", "willingness_gini", "score_mean", "score_std",
    "score_willingness_corr", "avg_bundle_size", "max_bundle_size",
    "task_candidate_gini", "courier_candidate_gini", "courier_ratio",
    "overlap_ratio", "remaining_time", "current_best", "assigned_ratio",
    "used_courier_ratio", "num_moves_tried", "last_gain", "budget",
)


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def featurize(features: Mapping[str, object], state: Mapping[str, object],
              action: Mapping[str, object]) -> Dict[str, float]:
    """Build a sparse numeric feature vector for (instance, state, action)."""
    merged = dict(features)
    merged.update(state)
    merged["budget"] = action.get("budget", 0.0)
    out: Dict[str, float] = {"bias": 1.0}
    for name in _NUMERIC_FEATURES:
        out[name] = _safe_float(merged.get(name))
    for name in ("size_class", "density_class", "willingness_class"):
        val = features.get(name)
        if val:
            out["%s=%s" % (name, val)] = 1.0
    move = action.get("move")
    if move:
        out["move=%s" % move] = 1.0
    params = action.get("params") or {}
    if isinstance(params, Mapping):
        for k, v in params.items():
            out["param.%s" % k] = _safe_float(v)
    return out


class PolicyModel:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.weights: Dict[str, float] = {}
        self.scale: float = 1.0
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.weights = {str(k): float(v)
                            for k, v in (data.get("weights") or {}).items()}
            self.scale = max(1e-9, float(data.get("scale", 1.0)))
        except Exception:
            self.weights = {}
            self.scale = 1.0

    def predict(self, features: Mapping[str, object], state: Mapping[str, object],
                action: Mapping[str, object]) -> float:
        if not self.weights:
            return 0.0
        x = featurize(features, state, action)
        y = sum(self.weights.get(k, 0.0) * v for k, v in x.items())
        return max(-1.0, min(1.0, y / self.scale))
