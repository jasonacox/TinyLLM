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


def test_models_endpoint_lists_pools_and_models_with_residency(fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(s.url, "llama3:8b")], "fast": [tr.Backend(s.url, "llama3.2:3b")]}, health_interval=999, request_timeout=5)
    with TestClient(tr.app) as c:
        data = {d["id"]: d for d in c.get("/v1/models").json()["data"]}
        assert data["chat"]["pool"] and data["chat"]["resident"] is True
        assert data["llama3.2:3b"]["resident"] is False and data["fast"]["healthy"] is True
        assert c.get("/pools").json()["chat"][0]["resident"] is True


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


def test_keep_alive_injected_for_ollama_backends_only(fake_ollama):
    o, v = fake_ollama(models=["llama3:8b"]), fake_ollama(models=["mistral"])
    tr.STATE.clear(); tr.STATE.update(pools={"chat": [tr.Backend(o.url, "llama3:8b", "ollama")], "big": [tr.Backend(v.url, "mistral", "openai")]},
                                      health_interval=999, request_timeout=5, keep_alive=-1)
    with TestClient(tr.app) as c:
        assert c.post("/v1/chat/completions", json={"model": "chat", "messages": []}).status_code == 200
        assert c.post("/v1/chat/completions", json={"model": "big", "messages": []}).status_code == 200
        with c.stream("POST", "/v1/chat/completions", json={"model": "chat", "stream": True, "messages": []}) as r:
            b"".join(r.iter_bytes())
        assert c.post("/v1/chat/completions", json={"model": "chat", "keep_alive": "5m", "messages": []}).status_code == 200
    o_posts = [c[2] for c in o.calls if c[0] == "POST" and c[1] == "/v1/chat/completions"]
    v_posts = [c[2] for c in v.calls if c[0] == "POST" and c[1] == "/v1/chat/completions"]
    assert [p.get("keep_alive") for p in o_posts] == [-1, -1, "5m"]              # injected (plain + stream); client value wins
    assert v_posts and all("keep_alive" not in p for p in v_posts)               # never sent to kind=openai


def test_keep_alive_wired_from_router_yaml(tmp_path, fake_ollama):
    o = fake_ollama()
    p = tmp_path / "r.yaml"; p.write_text(yaml.safe_dump({"pools": {"chat": [{"url": o.url, "model": "llama3:8b"}]}, "keep_alive": "30m", "health_interval": 999}))
    tr.STATE.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tr, "load_config", lambda path=str(p), _f=tr.load_config: _f(path))
        with TestClient(tr.app) as c:                                             # lifespan loads the config file
            assert c.post("/v1/chat/completions", json={"model": "chat", "messages": []}).status_code == 200
    assert [c[2].get("keep_alive") for c in o.calls if c[0] == "POST"] == ["30m"]


def test_load_config_example_parses_and_rejects_empty(tmp_path):
    cfg = tr.load_config(str(HERE / "router.yaml.example"))
    assert set(cfg["pools"]) == {"chat", "fast"} and cfg["pools"]["chat"][1].max_inflight == 2
    empty = tmp_path / "e.yaml"; empty.write_text("pools: {}\n")
    with pytest.raises(ValueError): tr.load_config(str(empty))


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
    assert "on_event" not in (HERE / "tinyrouter.py").read_text()               # lifespan, not the deprecated hook


def test_warm_cli_help_and_status_run(tmp_path, fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    pl = tmp_path / "placement.yaml"; pl.write_text(yaml.safe_dump({s.url: ["llama3:8b"]}))
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--help"], capture_output=True, text=True, timeout=30); assert r.returncode == 0
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--status", "--placement", str(pl)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "missing=[]" in r.stdout
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--placement", str(pl)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "llama3:8b=warm" in r.stdout
