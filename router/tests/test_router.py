"""TinyLLM Router tests — categories: unit, security, performance, retry, integration, functional, frame."""
import asyncio, json, os, subprocess, sys, time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tinyrouter as tr   # noqa: E402
import warm               # noqa: E402


def _b(url="http://x", model="m", healthy=True, resident=False, inflight=0, latency=1.0, cap=8):
    b = tr.Backend(url, model, "ollama", cap); b.healthy, b.resident, b.inflight, b.latency = healthy, resident, inflight, latency; return b


# ── unit ────────────────────────────────────────────────────────────────────
def test_pick_prefers_resident_then_least_loaded():
    cold_fast = _b("http://cold", resident=False, latency=0.1)
    warm_busy = _b("http://warm-busy", resident=True, inflight=3, latency=0.5)
    warm_idle = _b("http://warm-idle", resident=True, inflight=0, latency=0.9)
    assert tr.pick([cold_fast, warm_busy, warm_idle]).url == "http://warm-idle"      # (0+1)*0.9 < (3+1)*0.5
    assert tr.pick([cold_fast, _b("http://cold2", latency=0.2)]).url == "http://cold"  # no warm: least-loaded


def test_pick_respects_health_and_inflight_cap_and_exact_model():
    down = _b("http://down", healthy=False, resident=True)
    full = _b("http://full", resident=True, inflight=2, cap=2)
    ok = _b("http://ok", model="other")
    assert tr.pick([down, full, ok]).url == "http://ok"
    assert tr.pick([down, full, ok], model="m") is None
    assert tr.pick([], None) is None


def test_resolve_pool_vs_exact_model():
    pools = {"chat": [_b("http://a", model="llama3:8b"), _b("http://b", model="llama3:8b")], "fast": [_b("http://c", model="llama3.2:3b")]}
    bs, exact = tr.resolve(pools, "chat"); assert len(bs) == 2 and exact is None
    bs, exact = tr.resolve(pools, "llama3.2:3b"); assert [b.url for b in bs] == ["http://c"] and exact == "llama3.2:3b"
    bs, exact = tr.resolve(pools, "nope"); assert bs == [] and exact is None


def test_backend_observe_marks_unhealthy_after_two_failures():
    b = _b(); b.observe(1.0, ok=False); assert b.healthy; b.observe(1.0, ok=False); assert not b.healthy
    b.healthy = True; b.observe(0.2, ok=True); assert b.failures == 0 and 0.2 < b.latency < 1.0


def test_warm_status_and_pin_logic(fake_ollama):
    s = fake_ollama(models=["llama3:8b", "llama3.2:3b"], loaded=["llama3:8b"])
    assert warm.warm(s.url, "llama3:8b") == "warm"
    assert warm.warm(s.url, "llama3.2:3b").startswith("loaded in")
    assert warm.warm("http://127.0.0.1:1", "llama3:8b").startswith("fail:")
    assert any("missing=[]" in r for r in warm.status({s.url: ["llama3:8b", "llama3.2:3b"]}))


def test_load_config_reads_optional_keep_alive(tmp_path):
    p = tmp_path / "r.yaml"; p.write_text("pools:\n  chat:\n    - {url: http://a, model: m}\nkeep_alive: -1\n")
    assert tr.load_config(str(p))["keep_alive"] == -1
    p.write_text("pools:\n  chat:\n    - {url: http://a, model: m}\n")
    assert tr.load_config(str(p))["keep_alive"] is None


def test_warm_fetches_api_ps_once(fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    assert warm.warm(s.url, "llama3:8b") == "warm"
    assert sum(1 for c in s.calls if c[0] == "GET" and c[1] == "/api/ps") == 1


def test_ka_normalizes_keep_alive_values():
    assert tr._ka(-1) == -1 and tr._ka("-1") == -1 and tr._ka("0") == 0          # ints stay ints
    assert tr._ka("30m") == "30m" and tr._ka("1h") == "1h"                       # durations stay strings


# ── security ────────────────────────────────────────────────────────────────
def test_unknown_model_is_404_and_only_configured_models_are_reachable(fake_ollama):
    s = fake_ollama()
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        r = c.post("/v1/chat/completions", json={"model": "../../etc/passwd", "messages": []})
        assert r.status_code == 404
        r = c.post("/v1/chat/completions", json={"model": "llama3.2:3b", "messages": []})   # exists on the server, not in config
        assert r.status_code == 404
    assert not any(call[0] == "POST" for call in s.calls), "nothing was forwarded for unconfigured models"


def test_no_credentials_in_config_or_logs():
    src = (HERE / "tinyrouter.py").read_text() + (HERE / "warm.py").read_text()
    assert "Authorization" not in src and "api_key" not in src.lower()


# ── performance ─────────────────────────────────────────────────────────────
def test_pick_is_fast_over_many_backends():
    bs = [_b(f"http://b{i}", resident=(i % 7 == 0), inflight=i % 5, latency=0.1 + (i % 11) / 10) for i in range(2000)]
    t0 = time.time(); [tr.pick(bs) for _ in range(50)]; assert time.time() - t0 < 0.5


def test_acquire_without_queue_budget_is_immediate_none():
    at_cap = _b(resident=True, inflight=1, cap=1)
    tr.STATE.clear(); tr.STATE["queue_timeout"] = 0.0
    t0 = time.time()
    assert asyncio.run(tr._acquire([at_cap], None, [])) is None and time.time() - t0 < 0.2


def test_acquire_during_failover_never_waits():
    at_cap = _b(resident=True, inflight=1, cap=1)
    tr.STATE.clear(); tr.STATE["queue_timeout"] = 5.0                            # budget exists, but a failover pick ignores it
    t0 = time.time()
    assert asyncio.run(tr._acquire([at_cap], None, [at_cap])) is None and time.time() - t0 < 0.2


# ── retry / failover ────────────────────────────────────────────────────────
def test_failover_to_second_backend_when_first_errors(fake_ollama):
    bad, good = fake_ollama(fail=True), fake_ollama(reply="from-good")
    b1, b2 = tr.Backend(bad.url, "llama3:8b"), tr.Backend(good.url, "llama3:8b")
    b1.healthy = b2.healthy = True; b1.resident = True; b2.resident = False       # the warm one is broken
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [b1, b2]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:                                                 # startup sweep marks `bad` unhealthy+cold...
        b1.healthy, b1.resident = True, True                                      # ...force it back as the preferred (warm) pick
        r = c.post("/v1/chat/completions", json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from-good"
    assert b1.failures >= 1


def _stream_pool(*servers):
    bs = [tr.Backend(x.url, "llama3:8b") for x in servers]
    tr.STATE.clear(); tr.STATE.update(pools={"chat": bs}, health_interval=999, request_timeout=5)
    return bs


def test_stream_fails_over_when_first_backend_returns_5xx(fake_ollama):
    bad, good = fake_ollama(fail_stream=True, reply="bad"), fake_ollama(loaded=[], reply="from-good")
    b1, b2 = _stream_pool(bad, good)
    with TestClient(tr.app) as c:
        b1.resident, b2.resident = True, False                                    # the broken one is the preferred pick
        with c.stream("POST", "/v1/chat/completions", json={"model": "chat", "stream": True, "messages": []}) as r:
            body = b"".join(r.iter_bytes())
        assert r.status_code == 200 and b'"content": "from-good"' in body and b"[DONE]" in body
    assert any(c[1] == "/v1/chat/completions" for c in bad.calls if c[0] == "POST"), "bad backend was tried first"
    assert b1.failures == 1 and "500" in b1.last_error and b2.failures == 0
    assert b1.inflight == 0 and b2.inflight == 0, "inflight counters leaked"


def test_stream_all_backends_5xx_is_503_and_inflight_released(fake_ollama):
    b1, b2 = _stream_pool(fake_ollama(fail_stream=True), fake_ollama(fail_stream=True))
    with TestClient(tr.app) as c:
        r = c.post("/v1/chat/completions", json={"model": "chat", "stream": True, "messages": []})
    assert r.status_code == 503
    assert (b1.inflight, b2.inflight) == (0, 0) and b1.failures == 1 and b2.failures == 1


def test_stream_connection_error_fails_over_without_leaking(fake_ollama):
    good = fake_ollama(reply="from-good")
    dead = tr.Backend("http://127.0.0.1:1", "llama3:8b")
    b_good = tr.Backend(good.url, "llama3:8b")
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [dead, b_good]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        dead.healthy, dead.resident, b_good.resident = True, True, False
        with c.stream("POST", "/v1/chat/completions", json={"model": "chat", "stream": True, "messages": []}) as r:
            body = b"".join(r.iter_bytes())
    assert b'"content": "from-good"' in body and dead.inflight == 0 and b_good.inflight == 0


def test_all_backends_down_is_503(fake_ollama):
    s = fake_ollama(fail=True)
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        r = c.post("/v1/chat/completions", json={"model": "chat", "messages": []})
    assert r.status_code == 503


def test_embeddings_fails_over_to_second_backend(fake_ollama):
    bad, good = fake_ollama(fail=True), fake_ollama(models=["nomic-embed-text"], loaded=["nomic-embed-text"])
    b1, b2 = tr.Backend(bad.url, "nomic-embed-text"), tr.Backend(good.url, "nomic-embed-text")
    tr.STATE.clear(); tr.STATE.update(pools={"embed": [b1, b2]}, health_interval=999, request_timeout=5, queue_timeout=0.0, keep_alive=None)
    with TestClient(tr.app) as c:
        b1.healthy, b1.resident, b2.healthy, b2.resident = True, True, True, False   # the broken one is preferred
        r = c.post("/v1/embeddings", json={"model": "embed", "input": "x"})
    assert r.status_code == 200 and r.json()["data"][0]["embedding"]
    assert b1.failures >= 1 and b1.errors >= 1 and b1.inflight == 0 and b2.inflight == 0


# ── integration ─────────────────────────────────────────────────────────────
def test_health_sweep_reads_residency_from_api_ps(fake_ollama):
    s = fake_ollama(models=["llama3:8b", "llama3.2:3b"], loaded=["llama3.2:3b"])
    b_cold, b_warm = tr.Backend(s.url, "llama3:8b"), tr.Backend(s.url, "llama3.2:3b")
    async def go():
        async with httpx.AsyncClient() as client:
            await tr.check_backend(client, b_cold); await tr.check_backend(client, b_warm)
    asyncio.run(go())
    assert b_cold.healthy and not b_cold.resident
    assert b_warm.healthy and b_warm.resident


def test_residency_does_not_false_positive_on_latest_tag(fake_ollama):
    # Only qwen3:latest is loaded. A bare "qwen3" answers to it; a different tag (qwen3:8b) must NOT.
    s = fake_ollama(models=["qwen3", "qwen3:8b"], loaded=["qwen3:latest"])
    b_bare, b_tagged = tr.Backend(s.url, "qwen3"), tr.Backend(s.url, "qwen3:8b")
    async def go():
        async with httpx.AsyncClient() as client:
            await tr.check_backend(client, b_bare); await tr.check_backend(client, b_tagged)
    asyncio.run(go())
    assert b_bare.resident           # bare name matches its :latest
    assert not b_tagged.resident     # qwen3:8b is a different model, cold


def test_models_endpoint_lists_pools_and_models_with_residency(fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")], "fast": [tr.Backend(s.url, "llama3.2:3b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        data = {d["id"]: d for d in c.get("/v1/models").json()["data"]}
        assert data["chat"]["pool"] and data["chat"]["resident"] is True
        assert data["llama3.2:3b"]["resident"] is False and data["fast"]["healthy"] is True
        assert c.get("/pools").json()["chat"][0]["resident"] is True


def test_metrics_endpoint_exposes_prometheus_series(fake_ollama):
    o = fake_ollama(loaded=["llama3:8b"])
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(o.url, "llama3:8b")]}, health_interval=999, request_timeout=5, queue_timeout=0.0, keep_alive=None)
    with TestClient(tr.app) as c:
        c.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        m = c.get("/metrics")
    assert m.status_code == 200 and m.headers["content-type"].startswith("text/plain")
    assert "# TYPE tinyrouter_backend_requests_total counter" in m.text
    assert "# TYPE tinyrouter_backend_healthy gauge" in m.text
    assert 'tinyrouter_backend_requests_total{pool="chat"' in m.text


def test_sighup_reload_swaps_pools_and_keeps_live_state(tmp_path, fake_ollama):
    o1, o2 = fake_ollama(loaded=["llama3:8b"]), fake_ollama(loaded=["mistral"])
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"pools": {"chat": [{"url": o1.url, "model": "llama3:8b"}]}, "health_interval": 999}))
    tr.STATE.clear(); tr.STATE.update(tr.load_config(str(p)))
    b_old = tr.STATE["pools"]["chat"][0]; b_old.inflight, b_old.healthy, b_old.requests = 3, True, 7   # live state
    p.write_text(yaml.safe_dump({"pools": {"chat": [{"url": o1.url, "model": "llama3:8b"}, {"url": o2.url, "model": "mistral"}]},
                                 "health_interval": 7, "queue_timeout": 2.0}))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tr, "load_config", lambda path=str(p), _f=tr.load_config: _f(path))
        tr.reload_config()
    pool = tr.STATE["pools"]["chat"]
    assert len(pool) == 2 and tr.STATE["health_interval"] == 7 and tr.STATE["queue_timeout"] == 2.0
    assert pool[0] is b_old and pool[0].inflight == 3 and pool[0].requests == 7   # same object, live state preserved


# ── functional ──────────────────────────────────────────────────────────────
def test_chat_end_to_end_routes_to_resident_backend_and_streams(fake_ollama):
    cold, warm_srv = fake_ollama(loaded=[], reply="cold"), fake_ollama(loaded=["llama3:8b"], reply="warm")
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(cold.url, "llama3:8b"), tr.Backend(warm_srv.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        r = c.post("/v1/chat/completions", json={"model": "chat", "messages": [{"role": "user", "content": "hi"}]})
        assert r.json()["choices"][0]["message"]["content"] == "warm"           # the resident one answered
        assert not any(call[0] == "POST" and call[1] == "/v1/chat/completions" for call in cold.calls)
        with c.stream("POST", "/v1/chat/completions", json={"model": "llama3:8b", "stream": True, "messages": []}) as sr:
            body = b"".join(sr.iter_bytes())
        assert b"[DONE]" in body and b'"content": "warm"' in body


def test_keep_alive_pins_resident_ollama_models_natively(fake_ollama):
    # Ollama's OpenAI /v1 endpoint ignores a body keep_alive, so the router pins RESIDENT ollama models with a
    # native /api/generate during the health sweep — never force-loading a cold model, never touching kind=openai.
    o = fake_ollama(models=["llama3:8b"], loaded=["llama3:8b"])          # resident -> pinned
    cold = fake_ollama(models=["llama3:8b"], loaded=[])                  # not resident -> must NOT be force-loaded
    v = fake_ollama(models=["mistral"])                                 # openai kind -> never pinned
    b_res, b_cold, b_v = tr.Backend(o.url, "llama3:8b", "ollama"), tr.Backend(cold.url, "llama3:8b", "ollama"), tr.Backend(v.url, "mistral", "openai")
    async def go():
        async with httpx.AsyncClient() as client:
            for b in (b_res, b_cold, b_v):
                await tr.check_backend(client, b, keep_alive=-1)
    asyncio.run(go())
    gen = lambda s: [c for c in s.calls if c[0] == "POST" and c[1] == "/api/generate"]
    assert gen(o) and gen(o)[0][2].get("keep_alive") == -1              # resident ollama model pinned natively
    assert not gen(cold)                                               # cold model never force-loaded by a health check
    assert not gen(v)                                                  # kind=openai never gets a native pin


def test_keep_alive_not_injected_into_v1_requests(fake_ollama):
    # The no-op injection is gone: a configured keep_alive must NOT ride along on the /v1 chat payload.
    o = fake_ollama(models=["llama3:8b"], loaded=["llama3:8b"])
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(o.url, "llama3:8b", "ollama")]},
                                      health_interval=999, request_timeout=5, queue_timeout=0.0, keep_alive=-1)
    with TestClient(tr.app) as c:
        assert c.post("/v1/chat/completions", json={"model": "chat", "messages": []}).status_code == 200
    chat_posts = [c[2] for c in o.calls if c[0] == "POST" and c[1] == "/v1/chat/completions"]
    assert chat_posts and all("keep_alive" not in p for p in chat_posts)


def test_keep_alive_wired_from_router_yaml(tmp_path, fake_ollama):
    o = fake_ollama(loaded=["llama3:8b"])                                # resident so the first sweep pins it
    p = tmp_path / "r.yaml"; p.write_text(yaml.safe_dump({"pools": {"chat": [{"url": o.url, "model": "llama3:8b"}]}, "keep_alive": "30m", "health_interval": 999}))
    tr.STATE.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tr, "load_config", lambda path=str(p), _f=tr.load_config: _f(path))
        with TestClient(tr.app) as c:                                             # lifespan loads the config and sweeps once
            assert c.post("/v1/chat/completions", json={"model": "chat", "messages": []}).status_code == 200
    assert [c[2].get("keep_alive") for c in o.calls if c[0] == "POST" and c[1] == "/api/generate"] == ["30m"]


def test_load_config_example_parses_and_rejects_empty(tmp_path):
    cfg = tr.load_config(str(HERE / "router.yaml.example"))
    assert set(cfg["pools"]) == {"chat", "fast"} and cfg["pools"]["chat"][1].max_inflight == 2
    empty = tmp_path / "e.yaml"; empty.write_text("pools: {}\n")
    with pytest.raises(ValueError): tr.load_config(str(empty))


def test_embeddings_route_to_resident_pool_backend(fake_ollama):
    o = fake_ollama(models=["nomic-embed-text"], loaded=["nomic-embed-text"])
    tr.STATE.clear(); tr.STATE.update(pools={"embed": [tr.Backend(o.url, "nomic-embed-text")]}, health_interval=999, request_timeout=5, queue_timeout=0.0, keep_alive=None)
    with TestClient(tr.app) as c:
        r = c.post("/v1/embeddings", json={"model": "embed", "input": "hello"})
    assert r.status_code == 200 and r.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert any(call[0] == "POST" and call[1] == "/v1/embeddings" for call in o.calls)


def test_queue_waits_for_a_slot_then_picks():
    b = _b(resident=True, inflight=1, cap=1)                                     # at its cap right now
    tr.STATE.clear(); tr.STATE["queue_timeout"] = 1.0
    async def go():
        async def free_later():
            await asyncio.sleep(0.1); b.inflight = 0                             # a slot frees mid-wait
        t = asyncio.create_task(free_later())
        got = await tr._acquire([b], None, [])
        await t
        return got
    assert asyncio.run(go()) is b                                               # waited, then picked it


# ── frame ───────────────────────────────────────────────────────────────────
def test_app_starts_and_health_answers(fake_ollama):
    s = fake_ollama()
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        h = c.get("/health").json(); assert h["ok"] and h["backends"] == 1 and h["healthy"] == 1


def test_lifespan_starts_and_cancels_health_task(fake_ollama):
    s = fake_ollama()
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        task = tr.STATE["_health_task"]; assert not task.done()
        assert c.get("/health").json()["healthy"] == 1
    assert task.cancelled() or task.done()


def test_example_config_parses_queue_timeout_and_keep_alive(tmp_path):
    cfg = tr.load_config(str(HERE / "router.yaml.example"))
    assert cfg["queue_timeout"] == 0.0 and "keep_alive" in cfg          # both knobs present/defaulted, example valid


def test_metrics_output_is_well_formed_prometheus(fake_ollama):
    s = fake_ollama()
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        lines = [ln for ln in c.get("/metrics").text.splitlines() if ln]
    # every non-comment line is `name{labels} value`; every metric has a HELP and a TYPE comment
    helps = {ln.split()[2] for ln in lines if ln.startswith("# HELP")}
    types = {ln.split()[2] for ln in lines if ln.startswith("# TYPE")}
    samples = {ln.split("{")[0] for ln in lines if not ln.startswith("#")}
    assert samples and samples <= helps and samples <= types
    for ln in (x for x in lines if not x.startswith("#")):
        assert ln.endswith(tuple("0123456789")) and "{" in ln and ln.rstrip().split()[-1]   # value present
    assert "on_event" not in (HERE / "tinyrouter.py").read_text()               # lifespan, not the deprecated hook


def test_warm_cli_help_and_status_run(tmp_path, fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    pl = tmp_path / "placement.yaml"; pl.write_text(yaml.safe_dump({s.url: ["llama3:8b"]}))
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--help"], capture_output=True, text=True, timeout=30); assert r.returncode == 0
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--status", "--placement", str(pl)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "missing=[]" in r.stdout
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--placement", str(pl)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "llama3:8b=warm" in r.stdout
