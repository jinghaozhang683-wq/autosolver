"""LongCat advisor for weak strategy bias, not direct solving."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, Iterable, Mapping, Optional

_DEFAULT_BASE = "https://api.longcat.chat/openai/v1"
_DEFAULT_MODEL = "LongCat-2.0-Preview"


def _env_key() -> Optional[str]:
    for name in ("LONGCAT_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


class LlmAdvisor:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or _DEFAULT_BASE).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODEL
        self.api_key = api_key or _env_key()

    def available(self) -> bool:
        return bool(self.api_key)

    def advise(self, instance_features: Mapping[str, object],
               search_state: Mapping[str, object],
               move_history: Iterable[Mapping[str, object]],
               available_actions: Iterable[str],
               timeout: float = 4.0) -> Dict[str, object]:
        if not self.available():
            return {}
        prompt = self._prompt(instance_features, search_state,
                              list(move_history), list(available_actions))
        try:
            text = self._chat(prompt, timeout)
            return self._parse_json(text)
        except Exception:
            return {}

    def _chat(self, prompt: str, timeout: float) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900,
            "temperature": 0.1,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    def _prompt(self, features, state, history, actions) -> str:
        payload = {
            "instance_features": features,
            "search_state": state,
            "move_history": history[-8:],
            "available_actions": actions,
        }
        return (
            "You are an advisor for a courier-dispatch optimizer. Do not solve "
            "the assignment. Return strict JSON only with keys: diagnosis, "
            "action_bias, suggested_params, risk. action_bias maps available "
            "action names to small numeric biases in [-0.5, 0.5].\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    def _parse_json(self, text: str) -> Dict[str, object]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return {}
        bias = data.get("action_bias")
        if isinstance(bias, dict):
            clean = {}
            for k, v in bias.items():
                try:
                    clean[str(k)] = max(-0.5, min(0.5, float(v)))
                except Exception:
                    pass
            data["action_bias"] = clean
        return data
