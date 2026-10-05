#!/usr/bin/env python3
"""
warm.py — keep your models RESIDENT on every Ollama server, so a request never pays a model load.

Ollama unloads an idle model after 5 minutes (OLLAMA_KEEP_ALIVE default) and by default keeps at
most 3 loaded. A model that is not resident costs a load on the next request: seconds for a 3B,
30-60 s for a 30B. This script reads a *placement* (which models should live on which server),
and (re)asserts keep_alive=-1 for each one — an empty-prompt generate (or embed) call loads it if
cold and pins it if warm. Idempotent. Run it from cron every 10 minutes and the placement survives
server restarts.

    python3 warm.py                      # assert placement.yaml
    python3 warm.py --status             # what is loaded where vs. what should be
    python3 warm.py --placement my.yaml  # another placement file

placement.yaml:
    http://192.168.1.6:11434:  [llama3:8b, qwen3:30b-a3b, nomic-embed-text]
    http://192.168.1.7:11434:  [llama3.2:3b]

Pair with these per-server settings (see README.md): OLLAMA_KEEP_ALIVE=-1, OLLAMA_MAX_LOADED_MODELS=<n>,
and OLLAMA_CONTEXT_LENGTH sized to the box — a 262k-token default context is what makes an 8 GB model
"predicted" at 29 GiB and evict its neighbours.

Author: Jordan Koch (kochj23) with Claude, for TinyLLM (Jason Cox). MIT.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED_YEAR = "2319"                     # keep_alive=-1 shows as an expires_at ~300 years out


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def loaded_models(url):
    """{name: expires_at} for models resident on an Ollama server."""
    return {m["name"]: (m.get("expires_at") or "")[:19] for m in _get(f"{url}/api/ps").get("models", [])}


def warm(url, model, keep_alive=-1):
    """Load+pin one model. Returns 'warm', 'loaded in Ns', or 'fail: reason'."""
    try:
        was_warm = model in loaded_models(url) or f"{model}:latest" in loaded_models(url)
    except Exception as e:  # noqa: BLE001
        return f"fail: ps {e}"
    t0 = time.time()
    try:
        if "embed" in model:
            _post(f"{url}/api/embed", {"model": model, "input": "warm", "keep_alive": keep_alive})
        else:
            _post(f"{url}/api/generate", {"model": model, "prompt": "", "keep_alive": keep_alive})
    except Exception as e:  # noqa: BLE001
        return f"fail: {str(e)[:80]}"
    return "warm" if was_warm else f"loaded in {time.time() - t0:.0f}s"


def status(placement):
    rows = []
    for url, models in placement.items():
        try:
            loaded = loaded_models(url)
        except Exception as e:  # noqa: BLE001
            rows.append(f"{url}: unreachable ({e})"); continue
        missing = [m for m in models if m not in loaded and f"{m}:latest" not in loaded]
        unpinned = [m for m, exp in loaded.items() if (m in models or m.replace(":latest", "") in models) and exp and not exp.startswith(PINNED_YEAR)]
        extra = [m for m in loaded if m not in models and m.replace(":latest", "") not in models]
        rows.append(f"{url}: loaded={sorted(loaded)} missing={missing} unpinned={unpinned} extra={extra}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--placement", default=os.path.join(HERE, "placement.yaml"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--keep-alive", default="-1", help="-1 forever (default), or e.g. 30m")
    a = ap.parse_args(argv)
    with open(a.placement) as f:
        placement = yaml.safe_load(f) or {}
    if a.status:
        print("\n".join(status(placement))); return 0
    ka = int(a.keep_alive) if a.keep_alive.lstrip("-").isdigit() else a.keep_alive
    fails = 0
    for url, models in placement.items():
        results = {m: warm(url, m, ka) for m in models}
        print(f"{url}: " + ", ".join(f"{m}={r}" for m, r in results.items()))
        fails += sum(1 for r in results.values() if r.startswith("fail"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
