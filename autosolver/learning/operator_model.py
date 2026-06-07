"""Operator selector for learning-guided local search."""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, Mapping, Optional


class OperatorModel:
    def __init__(self, path: Optional[str] = None, seed=None):
        self.path = path
        self.rng = random.Random(seed)
        self.values = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.values = data.get("values") or {}
        except Exception:
            self.values = {}

    def predict(self, profile_key: str, state: Mapping[str, object],
                operator: str) -> float:
        key = "%s|%s" % (profile_key, operator)
        if key in self.values:
            try:
                return float(self.values[key])
            except Exception:
                return 0.0
        # Conservative defaults: repair has broader reach, add_backup is cheap.
        if operator == "drop_repair":
            return 0.05
        if operator == "add_backup":
            return 0.03
        return 0.0

    def choose(self, profile_key: str, state: Mapping[str, object],
               operators: Iterable[str]) -> str:
        ops = list(operators)
        if not ops:
            return ""
        scored = [(self.predict(profile_key, state, op), self.rng.random(), op)
                  for op in ops]
        scored.sort(reverse=True)
        return scored[0][2]
