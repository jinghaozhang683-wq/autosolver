"""LLM direct-reasoning strategy (OpenAI-compatible chat API).

Presents a compact view of the instance to a large language model and asks it to
propose an assignment, which is then parsed and validated by the evaluator. The
LLM rarely beats the exact solver on large instances, but it is one of the
strategies the agent is required to be able to try; the agent learns when it is
worth calling.

Defaults to Meituan **LongCat** (`https://api.longcat.chat/openai/v1`,
OpenAI-compatible), but works with any OpenAI-compatible endpoint. Configuration
via environment (never hard-code the key):

    LONGCAT_API_KEY / OPENAI_API_KEY / LLM_API_KEY   - the key
    LLM_BASE_URL                                     - override base url
    LLM_MODEL                                        - override model

If no key is set the strategy reports itself unavailable and the agent skips it.
Uses only the standard library (urllib) - no extra dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

from ..problem import Problem
from ..solution import Solution
from .base import Solver

_DEFAULT_BASE = "https://api.longcat.chat/openai/v1"
_DEFAULT_MODEL = "LongCat-2.0-Preview"


def _env_key() -> Optional[str]:
    for name in ("LONGCAT_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


class LlmSolver(Solver):
    name = "llm"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_candidates_per_task: int = 6, max_tasks: int = 40):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or _DEFAULT_BASE).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODEL
        self.api_key = api_key or _env_key()
        self.max_candidates_per_task = max_candidates_per_task
        self.max_tasks = max_tasks

    def available(self) -> bool:
        return bool(self.api_key)

    def solve(self, problem: Problem, time_budget: float,
              incumbent: Optional[Solution] = None) -> Solution:
        t0 = time.time()
        if not self.available() or problem.task_count > self.max_tasks:
            return self._finish(problem, [], t0)
        try:
            prompt = self._build_prompt(problem, incumbent)
            text = self._chat(prompt, timeout=max(3.0, time_budget - 0.2))
            groups = self._parse(problem, text)
            groups = self._repair(problem, groups)   # LLMs violate uniqueness; fix it
        except Exception:
            groups = []
        return self._finish(problem, groups, t0)

    def _repair(self, problem: Problem, groups):
        """LLMs routinely reuse couriers / double-cover tasks. Keep the model's
        non-conflicting picks, drop conflicts, then greedily cover any tasks it
        missed -- turning the LLM into a useful proposer."""
        used_cour = set()
        used_task = set()
        out = []
        for tk, cs in groups:
            tids = problem.bundle_tasks(tk)
            if any(t in used_task for t in tids):
                continue
            keep = [c for c in cs
                    if problem.info(tk, c) is not None and c not in used_cour]
            if not keep:
                continue
            out.append((tk, keep))
            used_cour.update(keep)
            used_task.update(tids)
        # greedily cover remaining tasks with the cheapest free single courier
        remaining = [t for t in problem.tasks if t not in used_task]
        if remaining:
            singles = [c for c in problem.candidates if c.task_count == 1]
            singles.sort(key=lambda c: c.cost)
            rem = set(remaining)
            for c in singles:
                t = c.task_ids[0]
                if t in rem and c.courier_id not in used_cour:
                    out.append((c.task_key, [c.courier_id]))
                    used_cour.add(c.courier_id)
                    rem.discard(t)
        return out

    # -- HTTP (OpenAI-compatible) ------------------------------------------
    def _chat(self, prompt: str, timeout: float) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.3,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    # -- prompt / parse ----------------------------------------------------
    def _build_prompt(self, problem: Problem, incumbent: Optional[Solution]) -> str:
        lines = []
        for task_key in sorted(problem.by_task_key):
            cands = problem.by_task_key[task_key]
            top = sorted(cands, key=lambda c: c.cost)[:self.max_candidates_per_task]
            opts = ", ".join("%s(s=%.1f,w=%.2f)" % (c.courier_id, c.score, c.willingness)
                             for c in top)
            lines.append("%s: %s" % (task_key, opts))
        body = "\n".join(lines)
        hint = ""
        if incumbent is not None and incumbent.feasible:
            hint = ("\nA known feasible solution scores %.2f (lower is better); "
                    "try to beat it.\n" % incumbent.score)
        return (
            "You are solving a courier dispatch optimisation. Assign every task "
            "(or bundle 'A,B') to one or more couriers. Each courier may be used "
            "at most once across the whole solution; each task covered at most "
            "once. Cost per group = p_complete*expected_score + "
            "fail_prob*100*num_tasks, where fail_prob = prod(1-w_i), "
            "p_complete = 1-fail_prob, expected_score = sum(w_i*s_i)/sum(w_i). "
            "Uncovered task costs 100. Minimise total cost.\n\n"
            "Candidates (task: courier(score,willingness) ...):\n"
            + body + hint +
            "\n\nReturn ONLY a JSON list of [task_key, [courier_ids]] pairs, "
            "e.g. [[\"T0001\", [\"C003\",\"C007\"]], [\"T0002,T0005\", [\"C010\"]]]. "
            "No prose, no markdown fences."
        )

    def _parse(self, problem: Problem, text: str) -> List[Tuple[str, List[str]]]:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        groups = []
        for item in data:
            try:
                tk, cs = item[0], item[1]
                if isinstance(cs, str):
                    cs = [cs]
                if tk in problem.by_task_key:
                    groups.append((tk, [str(c) for c in cs]))
            except Exception:
                continue
        return groups
