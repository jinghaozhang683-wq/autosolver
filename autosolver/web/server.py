"""Web UI for the AutoSolver agent.

    python -m autosolver.web.server [--port 5000]

Serves a single-page app that:
  * lets you pick a built-in case or paste your own,
  * runs the agent and **streams its decisions live** (Server-Sent Events) so you
    watch it explore strategies, evaluate, and pivot in real time,
  * visualises the final task -> courier assignment,
  * shows the per-strategy learned values and the cross-run memory.

The agent itself is unchanged; the server just wires its `on_event` callback to
an SSE stream.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

from flask import Flask, Response, request, send_from_directory

from ..problem import Problem
from ..agent import AutoSolverAgent
from ..memory import Memory

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "static")
_DATASETS = os.path.abspath(os.path.join(_HERE, "..", "..", "datasets"))
# Shared with the CLI / pre-training so the web reflects what the agent has
# learned (and keeps learning across web runs). Lives next to the package.
_MEMORY = os.path.abspath(os.path.join(_HERE, "..", "trained_memory.json"))
_LEARNING_LOG = os.path.abspath(os.path.join(_HERE, "..", "autosolver_learning.jsonl"))

app = Flask(__name__)


def _builtin_cases():
    out = []
    if os.path.isdir(_DATASETS):
        for fn in sorted(os.listdir(_DATASETS)):
            if fn.endswith(".txt"):
                out.append(fn)
    return out


@app.route("/")
def index():
    return send_from_directory(_STATIC, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(_STATIC, path)


@app.route("/api/cases")
def api_cases():
    return {"cases": _builtin_cases()}


@app.route("/api/memory")
def api_memory():
    return {"memory": Memory(_MEMORY).data}


@app.route("/api/reset_memory", methods=["POST"])
def api_reset_memory():
    try:
        if os.path.exists(_MEMORY):
            os.remove(_MEMORY)
    except Exception:
        pass
    return {"ok": True}


def _load_case(payload):
    text = payload.get("case_text")
    if text and text.strip():
        return text
    name = payload.get("case_name")
    if name:
        path = os.path.join(_DATASETS, os.path.basename(name))
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


@app.route("/api/solve", methods=["POST"])
def api_solve():
    payload = request.get_json(force=True, silent=True) or {}
    text = _load_case(payload)
    budget = float(payload.get("budget", 10.0))
    use_cpsat = bool(payload.get("cpsat", False))
    use_llm = bool(payload.get("llm", False))

    q: "queue.Queue" = queue.Queue()

    def cb(ev):
        q.put(ev)

    def run():
        try:
            problem = Problem.parse(text)
            if not problem.candidates:
                q.put({"kind": "error", "text": "no candidates parsed from input"})
            else:
                agent = AutoSolverAgent(use_cpsat=use_cpsat, use_llm=use_llm,
                                        memory_path=_MEMORY,
                                        learning_log_path=_LEARNING_LOG,
                                        on_event=cb)
                agent.solve(problem, budget)
        except Exception as e:  # pragma: no cover
            q.put({"kind": "error", "text": "agent error: %s" % e})
        finally:
            q.put({"kind": "end"})

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            ev = q.get()
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
            if ev.get("kind") == "end":
                break

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def main():
    import argparse
    p = argparse.ArgumentParser()
    # cloud platforms (Render/Railway) inject the port via $PORT and expect
    # the app to bind 0.0.0.0; fall back to local defaults otherwise.
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    p.add_argument("--host", default=os.environ.get("HOST",
                   "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    args = p.parse_args()
    print("AutoSolver web UI -> http://%s:%d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
