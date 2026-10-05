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


def test_warm_cli_help_and_status_run(tmp_path, fake_ollama):
    s = fake_ollama(loaded=["llama3:8b"])
    pl = tmp_path / "placement.yaml"; pl.write_text(yaml.safe_dump({s.url: ["llama3:8b"]}))
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--help"], capture_output=True, text=True, timeout=30); assert r.returncode == 0
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--status", "--placement", str(pl)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "missing=[]" in r.stdout
    r = subprocess.run([sys.executable, str(HERE / "warm.py"), "--placement", str(pl)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "llama3:8b=warm" in r.stdout
