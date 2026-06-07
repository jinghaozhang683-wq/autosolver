"""Train the contextual policy model from JSONL learning logs.

Usage:
    python -m autosolver.learning.train --log autosolver_learning.jsonl \
        --out autosolver_policy.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

from .policy_model import featurize


def _rows(path: str) -> Iterable[Tuple[Dict[str, float], float]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                result = row.get("result") or {}
                action = row.get("action") or {}
                x = featurize(row.get("features") or {},
                              row.get("state") or {}, action)
                y = float(result.get("improve_rate", 0.0))
                if math.isfinite(y):
                    yield x, y
            except Exception:
                continue


def _train_sgd(samples: List[Tuple[Dict[str, float], float]],
               l2: float, epochs: int, lr: float) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for _ in range(max(1, epochs)):
        for x, y in samples:
            pred = sum(w.get(k, 0.0) * v for k, v in x.items())
            err = pred - y
            for k, v in x.items():
                w[k] = w.get(k, 0.0) - lr * (err * v + l2 * w.get(k, 0.0))
    return w


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="autosolver_learning.jsonl")
    p.add_argument("--out", default="autosolver_policy.json")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args(argv)

    samples = list(_rows(args.log)) if os.path.exists(args.log) else []
    if not samples:
        print("no training samples found in %s" % args.log)
        return 1
    weights = _train_sgd(samples, args.l2, args.epochs, args.lr)
    scale = max(1.0, max(abs(y) for _, y in samples))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"weights": weights, "scale": scale, "samples": len(samples)},
                  f, ensure_ascii=False, indent=1)
    print("trained policy model -> %s (%d samples, %d weights)" % (
        args.out, len(samples), len(weights)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
