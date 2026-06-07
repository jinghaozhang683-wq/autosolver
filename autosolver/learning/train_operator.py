"""Train local-search operator values from JSONL logs.

Usage:
    python -m autosolver.learning.train_operator --log autosolver_learning.jsonl \
        --out autosolver_operator_model.json
"""

from __future__ import annotations

import argparse
import json
import os


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="autosolver_learning.jsonl")
    p.add_argument("--out", default="autosolver_operator_model.json")
    args = p.parse_args(argv)
    if not os.path.exists(args.log):
        print("no log found: %s" % args.log)
        return 1

    sums = {}
    counts = {}
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event_type") != "local_operator":
                continue
            op = row.get("operator")
            prof = row.get("profile_key", "")
            if not op:
                continue
            # Lower score is better, so negative delta is positive value.
            value = -float(row.get("delta_score", 0.0))
            key = "%s|%s" % (prof, op)
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        print("no local_operator samples found in %s" % args.log)
        return 1
    values = {k: sums[k] / counts[k] for k in counts}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"values": values, "counts": counts}, f,
                  ensure_ascii=False, indent=1)
    print("trained operator model -> %s (%d profile/operator values)" % (
        args.out, len(values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
