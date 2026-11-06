# TinyLLM Router — keep models resident, route to the warm one

Two small tools for running TinyLLM across more than one local LLM server without paying a model
load on every turn.

| Tool | What it does |
|------|--------------|
| `tinyrouter.py` | OpenAI-compatible front door (`/v1/chat/completions`, `/v1/models`). Pools of backends; health-checked; prefers backends where the model is **resident** (Ollama `/api/ps`); least-loaded within that; fails over on error/timeout; per-backend inflight cap; streaming passthrough. |
| `warm.py` | Reads a placement (which models live on which Ollama server) and re-asserts `keep_alive=-1` for each — loads cold ones, pins warm ones. Run from cron every 10 minutes. `--status` shows loaded / missing / unpinned per server. |

## The problem these solve

A chat turn on a home cluster is often slow not because of generation but because of a **model load**:

- Ollama unloads idle models after **5 minutes** (`OLLAMA_KEEP_ALIVE` default `5m`).
- Ollama keeps at most **3 models loaded** by default (`OLLAMA_MAX_LOADED_MODELS`), evicting the oldest.
- Models load with their **full default context** (Qwen3 family: 262,144 tokens). The KV cache for that
  context is counted up front, so an 18 GB model is "predicted" at 29 GiB and evicts its neighbours —
  `sched.go: model predicted to exceed available memory, evicting` in the Ollama log.
- A client that picks "the fastest-pinging server" lands on a different box each turn, usually one
  where the model just went cold.

Measured on an 8-node Apple Silicon + Linux fleet before/after: chat time-to-first-token p50 **31.8 s → ~9 s**,
with the remaining time being prompt prefill, not loading.

## Setup

```bash
pip install -r router/requirements.txt
cp router/router.yaml.example router/router.yaml        # pools -> backends
cp router/placement.yaml.example router/placement.yaml  # what to keep resident where
python3 router/warm.py --status                         # see what is loaded now
python3 router/warm.py                                  # pin the placement
uvicorn tinyrouter:app --app-dir router --host 0.0.0.0 --port 8000
```

Point the chatbot at the router: `OPENAI_API_BASE=http://<router>:8000/v1 LLM_MODEL=chat`.

> **Security:** the router binds `0.0.0.0` with **no authentication** by default — keep it behind your LAN/firewall,
> or bind it locally (`HOST=127.0.0.1 python3 router/tinyrouter.py`, or `--host 127.0.0.1` with uvicorn). It never
> forwards the client's `Authorization` header to backends.

Optional `keep_alive:` (top level of `router.yaml`, e.g. `-1` or `30m`) is injected into every request sent to a
`kind: ollama` backend, so whatever the router routes to stays resident; a client-supplied `keep_alive` wins, and
`kind: openai` backends never receive it. Omit it to leave Ollama's own default alone.

Cron (every 10 minutes keeps the placement true across Ollama restarts):

```
*/10 * * * * /usr/bin/python3 /path/to/TinyLLM/router/warm.py >> /var/log/tinyllm-warm.log 2>&1
```

## Per-server Ollama settings

Set these in the Ollama service environment (systemd drop-in on Linux; the LaunchDaemon/LaunchAgent
plist `EnvironmentVariables` on macOS — the Ollama.app menu-bar version does not read them reliably,
use the plain `ollama serve` binary as a service):

| Variable | Why | Suggested |
|----------|-----|-----------|
| `OLLAMA_KEEP_ALIVE=-1` | never unload on idle | always |
| `OLLAMA_MAX_LOADED_MODELS=N` | how many stay resident | 3 on 16–32 GB, more on big boxes (8 on 512 GB) |
| `OLLAMA_CONTEXT_LENGTH=N` | caps the KV cache reservation; the single biggest memory lever | 8192 on 16 GB, 16384 on 24–32 GB, 32768 on 64 GB+ |
| `OLLAMA_NUM_PARALLEL=N` | concurrent requests per model (each multiplies KV memory) | 1–2 |

A request that asks for a different `num_ctx` than the server default spawns a **new runner** — that is a
reload. Keep clients on the server default.

### Apple Silicon notes

Unified memory is shared by the OS, Ollama and anything else (an MLX server, a browser). Ollama's
"available" estimate is ~75% of RAM minus what other processes hold, so pin conservatively on 16 GB
machines (one 3B + one 8B at 8k context is comfortable; two 8Bs are not). The M-series Ultra/Max
machines can hold a whole fleet's worth of models — put the big and the persona models there and
route to it first (hardware order, not ping order: a fast ping from a small box does not mean fast
prefill).

## Endpoints

- `POST /v1/chat/completions` — OpenAI format; `model` is a pool name or an exact model. Streams if `stream: true`.
- `GET /v1/models` — pools and models with `resident` / `healthy` flags, so pickers can prefer warm ones.
- `GET /health` — `{ok, backends, healthy, resident}`.
- `GET /pools` — full per-backend state (inflight, latency EMA, last error).

## Tests

```bash
pip install -r router/requirements-test.txt
pytest router/tests -q
```

The tests spin up a fake Ollama server in-process; nothing touches a real model.
