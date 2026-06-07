"""Persistent cross-experiment memory.

Stores, per instance-profile key, how each strategy has performed: how often it
was tried, how often it produced the winning solution, and its average
*improvement rate* (objective improvement per second). The agent loads this at
start-up to bias its exploration toward strategies that historically did well on
similar instances, and writes it back after every solve -- so it genuinely
"learns from history" across runs.

Backed by a small JSON file (default ``autosolver_memory.json`` in the cwd).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict

_LOCK = threading.Lock()


class Memory:
    def __init__(self, path: str = "autosolver_memory.json"):
        self.path = path
        self.data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def save(self) -> None:
        try:
            with _LOCK:
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
        except Exception:
            pass

    # -- stats access ------------------------------------------------------
    def stats(self, key: str, move: str) -> dict:
        return self.data.get(key, {}).get(move, {
            "tries": 0, "wins": 0, "improve_rate": 0.0, "avg_time": 0.0})

    def prior_value(self, key: str, move: str, default: float) -> float:
        """Learned expected improvement-rate for (profile, move), or default if
        unseen. Blended with a confidence that grows with the number of tries."""
        s = self.stats(key, move)
        n = s["tries"]
        if n <= 0:
            return default
        conf = n / (n + 3.0)               # more tries -> trust history more
        return conf * s["improve_rate"] + (1 - conf) * default

    def win_rate(self, key: str, move: str) -> float:
        s = self.stats(key, move)
        return s["wins"] / s["tries"] if s["tries"] else 0.0

    # -- updates -----------------------------------------------------------
    def update(self, key: str, move: str, improve_rate: float, seconds: float,
               won: bool) -> None:
        bucket = self.data.setdefault(key, {})
        s = bucket.setdefault(move, {
            "tries": 0, "wins": 0, "improve_rate": 0.0, "avg_time": 0.0})
        n = s["tries"]
        s["improve_rate"] = (s["improve_rate"] * n + improve_rate) / (n + 1)
        s["avg_time"] = (s["avg_time"] * n + seconds) / (n + 1)
        s["tries"] = n + 1
        if won:
            s["wins"] += 1
