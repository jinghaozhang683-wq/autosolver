"""Instance profiling: turn a Problem into a small set of features and a
discrete *profile key* used to look up / store learned strategy performance.

The agent treats instances with the same key as "similar", so what it learns on
one transfers to the next (cross-experiment learning).
"""

from __future__ import annotations

from typing import Dict

from .problem import Problem


def _size_class(nt: int) -> str:
    if nt <= 8:
        return "tiny"
    if nt <= 16:
        return "small"
    if nt <= 24:
        return "medium"
    if nt <= 35:
        return "large"
    return "xlarge"


def _density_class(cands_per_task: float) -> str:
    if cands_per_task < 30:
        return "sparse"
    if cands_per_task < 300:
        return "mid"
    return "dense"


def _willingness_class(mean_w: float) -> str:
    if mean_w < 0.25:
        return "lowwill"
    if mean_w < 0.6:
        return "midwill"
    return "highwill"


def _gini(values):
    """Gini coefficient of a list of values (0 = equal, 1 = unequal)."""
    n = len(values)
    if n < 2:
        return 0.0
    v = sorted(values)
    total = sum(v)
    if total < 1e-12:
        return 0.0
    cum = 0.0
    g = 0.0
    for i, x in enumerate(v, 1):
        cum += x
        g += i * x
    return (2.0 * g) / (n * total) - (n + 1.0) / n


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, var ** 0.5


def _corr(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, sx = _mean_std(xs)
    my, sy = _mean_std(ys)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


def profile(problem: Problem) -> Dict[str, object]:
    nt = max(1, problem.task_count)
    ncand = len(problem.candidates)

    # -- basic features
    cands_per_task = ncand / nt
    pair_ratio = (sum(1 for c in problem.candidates if c.task_count >= 2)
                  / max(1, ncand))
    mean_w = (sum(c.willingness for c in problem.candidates) / max(1, ncand))

    # -- bundle overlap: fraction of task pairs that co-appear in >1 bundle
    task_ids = list(problem.tasks)
    tid_index = {t: i for i, t in enumerate(task_ids)}
    co_occur = {}
    for c in problem.candidates:
        for i in range(len(c.task_ids)):
            for j in range(i + 1, len(c.task_ids)):
                key = (tid_index.get(c.task_ids[i], -1),
                       tid_index.get(c.task_ids[j], -1))
                co_occur[key] = co_occur.get(key, 0) + 1
    overlap_count = sum(1 for v in co_occur.values() if v > 1)
    total_pairs = max(1, nt * (nt - 1) // 2)
    overlap_ratio = overlap_count / total_pairs

    # -- willingness dispersion: Gini coefficient
    w_values = [c.willingness for c in problem.candidates]
    w_gini = _gini(w_values)
    scores = [c.score for c in problem.candidates]
    score_mean, score_std = _mean_std(scores)
    bundle_sizes = [c.task_count for c in problem.candidates]
    avg_bundle_size, _ = _mean_std(bundle_sizes)
    max_bundle_size = max(bundle_sizes) if bundle_sizes else 0
    task_candidate_counts = [len(problem.by_task_key.get(t, []))
                             for t in problem.by_task_key]
    courier_counts = {}
    for c in problem.candidates:
        courier_counts[c.courier_id] = courier_counts.get(c.courier_id, 0) + 1

    # -- courier-to-task ratio
    courier_ratio = problem.courier_count / nt

    feats = {
        "task_count": nt,
        "courier_count": problem.courier_count,
        "candidates": ncand,
        "candidate_count": ncand,
        "cands_per_task": cands_per_task,
        "density": cands_per_task / max(1, problem.courier_count),
        "pair_ratio": pair_ratio,
        "avg_willingness": mean_w,
        "willingness_mean": mean_w,
        "willingness_gini": round(w_gini, 4),
        "score_mean": round(score_mean, 4),
        "score_std": round(score_std, 4),
        "score_willingness_corr": round(_corr(scores, w_values), 4),
        "avg_bundle_size": round(avg_bundle_size, 4),
        "max_bundle_size": max_bundle_size,
        "task_candidate_gini": round(_gini(task_candidate_counts), 4),
        "courier_candidate_gini": round(_gini(list(courier_counts.values())), 4),
        "courier_ratio": round(courier_ratio, 2),
        "overlap_ratio": round(overlap_ratio, 4),
        "size_class": _size_class(nt),
        "density_class": _density_class(cands_per_task),
        "willingness_class": _willingness_class(mean_w),
    }

    # Keep the 3-part key for backward-compatible memory lookup.
    # Extra features (gini, overlap, courier_ratio) are still available for
    # the agent to use in logic, but the key stays coarse so each bucket
    # accumulates enough samples for the cross-run learning to be meaningful.
    feats["key"] = "%s/%s/%s" % (
        feats["size_class"], feats["density_class"], feats["willingness_class"])
    return feats
