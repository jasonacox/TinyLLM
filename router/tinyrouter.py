#!/usr/bin/env python3
"""
TinyLLM Router — an OpenAI-compatible front door for several local LLM servers.

Point your client (the TinyLLM chatbot, curl, any OpenAI SDK) at the router instead of at one
Ollama/vLLM/llama.cpp server. The router keeps a *pool* per model class and, for every request:

  1. uses only backends that passed the last health check,
  2. prefers backends where the requested model is RESIDENT (Ollama /api/ps) — a resident model
     answers immediately, a cold one costs a load of seconds to a minute,
  3. among those, picks the least-loaded by (inflight+1) × recent latency,
  4. fails over to the next backend if the chosen one errors or times out.

Why: on a multi-node home setup the slow part of a chat turn is usually not generation but a
model (re)load — Ollama unloads idle models after 5 minutes by default, and a naive client that
hops between servers pays that load again and again. See README.md for the pairing with warm.py.

    pip install fastapi uvicorn httpx pyyaml
    cp router.yaml.example router.yaml      # edit pools/backends
    uvicorn tinyrouter:app --host 0.0.0.0 --port 8000

Config (router.yaml):
    pools:
      chat:                                  # request "model": "chat" -> this pool
        - {url: http://192.168.1.6:11434,  model: llama3:8b, kind: ollama}
        - {url: http://192.168.1.77:11434, model: llama3:8b, kind: ollama, max_inflight: 2}
      fast:
        - {url: http://192.168.1.7:11434,  model: llama3.2:3b, kind: ollama}
    health_interval: 15        # seconds between health sweeps
    request_timeout: 120       # seconds per backend attempt
    keep_alive: -1             # optional: injected into requests to kind=ollama backends (unless the client set one)

A request may name a pool ("chat") or an exact model ("llama3:8b"); exact names route to every
pool backend serving that model. Only models in the config are reachable through the router.

Author: Jordan Koch (kochj23) with Claude, for TinyLLM (Jason Cox). MIT, same as TinyLLM.
"""
import asyncio
import json
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

log = logging.getLogger("tinyrouter")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = os.environ.get("ROUTER_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "router.yaml"))


def _model_key(name: str) -> str:
    """Ollama treats a bare model name as its :latest tag ("llama3" == "llama3:latest"). Match on a
    normalized key so a loaded "qwen3:latest" does not falsely mark "qwen3:8b" / "qwen3:32b" resident."""
    return name if ":" in name else f"{name}:latest"


class Backend:
    """One server+model pair inside a pool, with live health/residency/load state."""

    def __init__(self, url: str, model: str, kind: str = "ollama", max_inflight: int = 8):
        self.url = url.rstrip("/")
        self.model = model
        self.kind = kind                       # "ollama" (native API available) or "openai" (only /v1)
        self.max_inflight = int(max_inflight)
        self.healthy = False
        self.resident = False                  # model currently loaded (ollama only; openai kinds count as resident)
        self.inflight = 0
        self.latency = 1.0                     # EMA of seconds per request
        self.failures = 0
        self.last_error = ""
        self.requests = 0                      # lifetime counters (survive a SIGHUP reload; exposed at /metrics)
        self.errors = 0

    def score(self) -> float:
        """Lower is better: least-loaded, latency-aware. Residency is handled before scoring."""
        return (self.inflight + 1) * self.latency

    def observe(self, seconds: float, ok: bool):
        if ok:
            self.latency = 0.7 * self.latency + 0.3 * max(seconds, 0.05)
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= 2:
                self.healthy = False

    def as_dict(self) -> dict:
        return {"url": self.url, "model": self.model, "kind": self.kind, "healthy": self.healthy,
                "resident": self.resident, "inflight": self.inflight, "latency_s": round(self.latency, 3),
                "failures": self.failures, "requests": self.requests, "errors": self.errors,
                "last_error": self.last_error[:120]}


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    pools = {}
    for name, items in (cfg.get("pools") or {}).items():
        pools[name] = [Backend(b["url"], b["model"], b.get("kind", "ollama"), b.get("max_inflight", 8)) for b in items]
    if not pools:
        raise ValueError(f"{path}: no pools defined")
    return {"pools": pools, "health_interval": float(cfg.get("health_interval", 15)),
            "request_timeout": float(cfg.get("request_timeout", 120)),
            "queue_timeout": float(cfg.get("queue_timeout", 0.0)),   # >0: briefly wait for a slot before 503
            "keep_alive": cfg.get("keep_alive")}          # None = leave Ollama's own keep_alive alone


def pick(backends: list, model: Optional[str] = None) -> Optional[Backend]:
    """Resident + healthy + under its inflight cap first, then any healthy under cap; least-loaded within each.
    `model` narrows to backends serving that exact model (None = whole pool)."""
    cands = [b for b in backends if b.healthy and b.inflight < b.max_inflight and (model is None or b.model == model)]
    if not cands:
        return None
    warm = [b for b in cands if b.resident]
    return min(warm or cands, key=Backend.score)


def resolve(pools: dict, requested: str) -> tuple:
    """A pool name -> (pool backends, None). An exact model name -> (all backends with that model, model)."""
    if requested in pools:
        return pools[requested], None
    matches = [b for bs in pools.values() for b in bs if b.model == requested]
    return matches, (requested if matches else None)


def _ka(keep_alive):
    """Normalize a keep_alive config value: -1/0/30 stay ints, '30m'/'1h' stay strings."""
    return int(keep_alive) if str(keep_alive).lstrip("-").isdigit() else keep_alive


async def check_backend(client: httpx.AsyncClient, b: Backend, keep_alive=None):
    """Health + residency. Ollama: /api/tags must answer and /api/ps tells what is loaded. Others: /v1/models.
    When keep_alive is set, re-pin already-resident Ollama models with a native empty generate — Ollama's
    OpenAI /v1 endpoint ignores a body keep_alive, so per-request injection there is a no-op. We only pin what
    is already loaded (never force-load a cold model during a health check)."""
    try:
        if b.kind == "ollama":
            r = await client.get(f"{b.url}/api/tags", timeout=5)
            r.raise_for_status()
            ps = await client.get(f"{b.url}/api/ps", timeout=5)
            names = (m.get("name") or m.get("model") for m in ps.json().get("models", [])) if ps.status_code == 200 else ()
            loaded = {_model_key(n) for n in names if n}
            b.resident = _model_key(b.model) in loaded
        else:
            r = await client.get(f"{b.url}/v1/models", timeout=5)
            r.raise_for_status()
            b.resident = True                  # a vLLM/llama.cpp server holds its one model for its lifetime
        b.healthy = True
        b.failures = 0
        b.last_error = ""
    except Exception as e:  # noqa: BLE001 — any failure means "don't route here"
        b.healthy = False
        b.resident = False
        b.last_error = f"{type(e).__name__}: {e}"
        return
    if b.kind == "ollama" and keep_alive is not None and b.resident:
        try:
            await client.post(f"{b.url}/api/generate", json={"model": b.model, "prompt": "", "keep_alive": _ka(keep_alive)}, timeout=30)
        except Exception as e:  # noqa: BLE001 — best-effort pin; a pin failure never fails the health check
            log.debug("keep_alive pin on %s failed: %s", b.url, e)


async def _sweep(client, state):
    await asyncio.gather(*(check_backend(client, b, state.get("keep_alive"))
                           for bs in state["pools"].values() for b in bs))


async def health_loop(state: dict):
    async with httpx.AsyncClient() as client:
        while True:                                        # lifespan did the first sweep; wait before the next
            await asyncio.sleep(state["health_interval"])
            await _sweep(client, state)


STATE: dict = {}


def reload_config():
    """SIGHUP: re-read the config file and swap pools in place. A backend that persists (same url+model) keeps
    its live object, so its in-flight count and latency survive; new backends are added, removed ones are dropped
    (in-flight requests hold their own reference and finish unaffected). A bad file is logged and ignored."""
    try:
        new = load_config()
    except Exception as e:  # noqa: BLE001
        log.error("SIGHUP reload failed, keeping current config: %s", e)
        return
    old = {(b.url, b.model): b for bs in STATE["pools"].values() for b in bs}
    for bs in new["pools"].values():
        for i, b in enumerate(bs):
            prev = old.get((b.url, b.model))
            if prev:                                       # reuse the live object so inflight accounting stays correct
                prev.max_inflight, prev.kind = b.max_inflight, b.kind
                bs[i] = prev
    for k in ("pools", "health_interval", "request_timeout", "queue_timeout", "keep_alive"):
        STATE[k] = new[k]
    log.info("SIGHUP reloaded: pools=%s", {k: len(v) for k, v in STATE["pools"].items()})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not STATE:
        STATE.update(load_config())
    async with httpx.AsyncClient() as client:              # first sweep before serving
        await _sweep(client, STATE)
    task = STATE["_health_task"] = asyncio.create_task(health_loop(STATE))
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGHUP, reload_config)
    except (NotImplementedError, ValueError, AttributeError, RuntimeError):
        pass                                               # no SIGHUP (Windows) or not the main thread (e.g. TestClient)
    log.info("router up: pools=%s", {k: len(v) for k, v in STATE["pools"].items()})
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="TinyLLM Router", lifespan=lifespan)


@app.get("/health")
async def health():
    total = sum(len(v) for v in STATE["pools"].values())
    healthy = sum(1 for v in STATE["pools"].values() for b in v if b.healthy)
    return {"ok": healthy > 0, "backends": total, "healthy": healthy,
            "resident": sum(1 for v in STATE["pools"].values() for b in v if b.resident)}


@app.get("/pools")
async def pools():
    return {name: [b.as_dict() for b in bs] for name, bs in STATE["pools"].items()}


@app.get("/v1/models")
async def models():
    """Pools and exact models reachable through the router, with residency so pickers can prefer warm ones."""
    data, seen = [], set()
    for name, bs in STATE["pools"].items():
        data.append({"id": name, "object": "model", "owned_by": "tinyrouter", "pool": True,
                     "resident": any(b.resident for b in bs), "healthy": any(b.healthy for b in bs)})
        for b in bs:
            if b.model not in seen:
                seen.add(b.model)
                same = [x for xs in STATE["pools"].values() for x in xs if x.model == b.model]
                data.append({"id": b.model, "object": "model", "owned_by": "tinyrouter", "pool": False,
                             "resident": any(x.resident for x in same), "healthy": any(x.healthy for x in same)})
    return {"object": "list", "data": data}


async def _acquire(backends: list, exact: Optional[str], tried: list) -> Optional[Backend]:
    """Pick an untried, healthy, under-cap backend. On the FIRST dispatch (nothing tried yet), if the only
    reason nothing is pickable is that every healthy match is at its inflight cap, wait up to queue_timeout
    for a slot to free — smoothing a burst into a short queue instead of an immediate 503. Failover picks
    (tried non-empty) never wait."""
    deadline = time.monotonic() + STATE.get("queue_timeout", 0.0)
    while True:
        b = pick([x for x in backends if x not in tried], exact)
        if b is not None or tried:
            return b
        healthy = [x for x in backends if x.healthy and (exact is None or x.model == exact)]
        if not healthy or time.monotonic() >= deadline:     # nothing to wait for, or budget spent
            return None
        await asyncio.sleep(0.05)


async def _proxy(path: str, body: dict, requested: str, allow_stream: bool):
    """Route one OpenAI-style POST (chat or embeddings) to the best backend in the pool, failing over to the
    next on a 5xx or connection error. `path` is the backend path; streaming applies only to chat."""
    backends, exact = resolve(STATE["pools"], requested)
    if not backends:
        raise HTTPException(404, f"unknown model or pool '{requested}' — see /v1/models")
    tried = []
    while len(tried) < len(backends):
        b = await _acquire(backends, exact, tried)
        if b is None:
            break
        tried.append(b)
        payload = dict(body, model=b.model)
        b.inflight += 1
        b.requests += 1
        t0 = time.time()
        try:
            if allow_stream and payload.get("stream"):
                resp = await _stream(b, payload, t0, path)
                if resp is None:                               # 5xx on connect: already cleaned up, try the next one
                    b.errors += 1
                    continue
                return resp
            async with httpx.AsyncClient(timeout=STATE["request_timeout"]) as client:
                r = await client.post(f"{b.url}{path}", json=payload)
            b.inflight -= 1
            if r.status_code >= 500:
                b.errors += 1
                b.observe(time.time() - t0, ok=False)
                b.last_error = f"HTTP {r.status_code}"
                log.warning("backend %s -> HTTP %s, failing over", b.url, r.status_code)
                continue
            b.observe(time.time() - t0, ok=True)
            b.resident = True                                  # it just answered; it is loaded now
            return JSONResponse(r.json(), status_code=r.status_code)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            b.inflight -= 1
            b.errors += 1
            b.observe(time.time() - t0, ok=False)
            b.last_error = f"{type(e).__name__}: {e}"
            log.warning("backend %s failed (%s), failing over", b.url, b.last_error)
    raise HTTPException(503, f"no healthy backend answered for '{requested}' (tried {len(tried)})")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    return await _proxy("/v1/chat/completions", body, str(body.get("model") or ""), allow_stream=True)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Same pool/resident/failover routing as chat, for embedding models (RAG). Never streams."""
    body = await request.json()
    return await _proxy("/v1/embeddings", body, str(body.get("model") or ""), allow_stream=False)


@app.get("/metrics")
async def metrics():
    """Prometheus text exposition of per-backend state — no client library, just the counters we already keep."""
    series = {
        "tinyrouter_backend_healthy": ("gauge", "1 if the backend passed its last health check", lambda b: int(b.healthy)),
        "tinyrouter_backend_resident": ("gauge", "1 if the model is loaded on the backend", lambda b: int(b.resident)),
        "tinyrouter_backend_inflight": ("gauge", "in-flight requests on the backend", lambda b: b.inflight),
        "tinyrouter_backend_latency_seconds": ("gauge", "EMA of seconds per request", lambda b: round(b.latency, 3)),
        "tinyrouter_backend_requests_total": ("counter", "requests routed to the backend", lambda b: b.requests),
        "tinyrouter_backend_errors_total": ("counter", "requests to the backend that errored or 5xx'd", lambda b: b.errors),
    }
    out = []
    for metric, (mtype, help_, fn) in series.items():
        out.append(f"# HELP {metric} {help_}")
        out.append(f"# TYPE {metric} {mtype}")
        for name, bs in STATE["pools"].items():
            for b in bs:
                lbl = f'pool="{name}",url="{b.url}",model="{b.model}"'
                out.append(f"{metric}{{{lbl}}} {fn(b)}")
    return PlainTextResponse("\n".join(out) + "\n")


async def _stream(b: Backend, payload: dict, t0: float, path: str = "/v1/chat/completions") -> Optional[StreamingResponse]:
    """Open a streaming request to `b`. Returns the StreamingResponse, or None if the backend answered 5xx
    (response closed, inflight released, failure recorded) so _proxy() can fail over. Connection errors are
    re-raised as httpx errors after cleanup; _proxy()'s except releases inflight for those."""
    client = httpx.AsyncClient(timeout=STATE["request_timeout"])
    try:
        req = client.build_request("POST", f"{b.url}{path}", json=payload)
        resp = await client.send(req, stream=True)
    except BaseException:
        await client.aclose()
        raise
    if resp.status_code >= 500:
        await resp.aclose(); await client.aclose()
        b.inflight -= 1
        b.observe(time.time() - t0, ok=False)
        b.last_error = f"HTTP {resp.status_code} on stream"
        log.warning("backend %s -> HTTP %s on stream, failing over", b.url, resp.status_code)
        return None

    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
            b.observe(time.time() - t0, ok=True); b.resident = True
        except Exception:  # noqa: BLE001
            b.observe(time.time() - t0, ok=False)
            raise
        finally:
            b.inflight -= 1
            await resp.aclose(); await client.aclose()
    return StreamingResponse(gen(), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type", "text/event-stream"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
