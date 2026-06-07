"""JSONL learning trace writer."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Mapping, Optional

_LOCK = threading.Lock()


class LearningLogger:
    """Append state-action-result samples for offline policy training."""

    def __init__(self, path: Optional[str]):
        self.path = path

    def write(self, event: Mapping[str, object]) -> None:
        if not self.path:
            return
        row = dict(event)
        row.setdefault("timestamp", time.time())
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            with _LOCK:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass
