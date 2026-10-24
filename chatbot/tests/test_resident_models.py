"""Chatbot resident-model awareness — categories: unit, security, performance, retry, integration, functional, frame.
Run: pytest chatbot/tests -q   (the fake Ollama server lives in chatbot/tests/conftest.py)."""
import os, subprocess, sys, time
from pathlib import Path

import pytest

CHATBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")      # never a real server during import
os.environ.setdefault("PROMPT_FILE", str(CHATBOT / ".pytest-prompts.json"))
from conftest import FakeOllama   # noqa: E402
from app.core import llm           # noqa: E402


@pytest.fixture
def ollama(monkeypatch):
    s = FakeOllama(models=["llama3:8b", "llama3.2:3b", "nomic-embed-text:latest"], loaded=["llama3:8b"])
    monkeypatch.setattr(llm, "API_BASE", s.url + "/v1"); monkeypatch.setattr(llm, "OLLAMA_API", None)
    llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    yield s
    s.close()


# ── unit ────────────────────────────────────────────────────────────────────
def test_ollama_api_base_derivation(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_API", None)
    monkeypatch.setattr(llm, "API_BASE", "http://host:11434/v1"); assert llm.ollama_api_base() == "http://host:11434"
    monkeypatch.setattr(llm, "API_BASE", "http://host:11434/v1/"); assert llm.ollama_api_base() == "http://host:11434"
    monkeypatch.setattr(llm, "OLLAMA_API", "http://other:11434/"); assert llm.ollama_api_base() == "http://other:11434"


def test_get_models_detail_marks_resident_and_sorts_first(ollama, monkeypatch):
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3.2:3b", "llama3:8b", "nomic-embed-text:latest"])
    d = llm.get_models_detail()
    assert d[0] == {"id": "llama3:8b", "resident": True}
    assert {x["id"]: x["resident"] for x in d} == {"llama3:8b": True, "llama3.2:3b": False, "nomic-embed-text:latest": False}


def test_latest_tag_equivalence(ollama, monkeypatch):
    ollama.loaded = {"llama3:latest"}; llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3", "llama3:latest"])
    assert all(x["resident"] for x in llm.get_models_detail())


def test_latest_does_not_mark_other_tags_resident(ollama, monkeypatch):
    ollama.loaded = {"qwen3:latest"}; llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    monkeypatch.setattr(llm, "get_models", lambda: ["qwen3", "qwen3:8b", "qwen3:32b"])
    detail = {x["id"]: x["resident"] for x in llm.get_models_detail()}
    assert detail == {"qwen3": True, "qwen3:8b": False, "qwen3:32b": False}


def test_bare_name_loaded_matches_latest(ollama, monkeypatch):
    ollama.loaded = {"llama3"}; llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3:latest", "llama3:8b"])
    detail = {x["id"]: x["resident"] for x in llm.get_models_detail()}
    assert detail == {"llama3:latest": True, "llama3:8b": False}


# ── security ────────────────────────────────────────────────────────────────
def test_residency_calls_carry_no_api_key_and_stay_on_ollama_host(ollama):
    llm.resident_models()
    assert all(c[0] == "GET" and c[1] == "/api/ps" for c in ollama.calls)
    import inspect
    src = inspect.getsource(llm.resident_models) + inspect.getsource(llm.warm_model) + inspect.getsource(llm.ollama_api_base)
    assert "API_KEY" not in src and "Authorization" not in src       # residency/pin calls never carry the OpenAI key


# ── performance ─────────────────────────────────────────────────────────────
def test_resident_models_is_cached(ollama):
    llm.resident_models(); llm.resident_models(); llm.resident_models()
    assert sum(1 for c in ollama.calls if c[1] == "/api/ps") == 1
    t0 = time.time(); [llm.resident_models() for _ in range(200)]; assert time.time() - t0 < 0.2


# ── retry / resilience ──────────────────────────────────────────────────────
def test_non_ollama_backend_degrades_to_unknown(monkeypatch):
    monkeypatch.setattr(llm, "API_BASE", "http://127.0.0.1:1/v1"); monkeypatch.setattr(llm, "OLLAMA_API", None)
    llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    assert llm.resident_models() is None
    monkeypatch.setattr(llm, "get_models", lambda: ["gpt-4o"])
    assert llm.get_models_detail() == [{"id": "gpt-4o", "resident": None}]


def test_warm_model_is_noop_without_keep_alive_and_pins_with_it(ollama, monkeypatch):
    monkeypatch.setattr(llm, "KEEP_ALIVE", None); assert llm.warm_model("llama3.2:3b") is False
    monkeypatch.setattr(llm, "KEEP_ALIVE", "-1"); assert llm.warm_model("llama3.2:3b") is True
    post = [c for c in ollama.calls if c[0] == "POST"][-1]
    assert post[1] == "/api/generate" and post[2]["keep_alive"] == -1 and post[2]["model"] == "llama3.2:3b"
    monkeypatch.setattr(llm, "KEEP_ALIVE", "30m"); llm.warm_model("llama3:8b")
    assert [c for c in ollama.calls if c[0] == "POST"][-1][2]["keep_alive"] == "30m"
    assert llm.warm_model("llama3:8b", keep_alive="-1") is True
    monkeypatch.setattr(llm, "API_BASE", "http://127.0.0.1:1/v1"); assert llm.warm_model("llama3:8b") is False   # down -> False, no raise


# ── performance: nothing on the event loop waits for Ollama ──────────────────
async def _ticks_during(coro, period=0.02):
    """Run `coro` while a ticker coroutine counts; returns (result, ticks, seconds). A blocked loop -> ~0 ticks."""
    import asyncio
    ticks, done = 0, False
    async def ticker():
        nonlocal ticks
        while not done:
            await asyncio.sleep(period); ticks += 1
    t = asyncio.create_task(ticker()); t0 = time.time()
    try:
        res = await coro
    finally:
        done = True; await t
    return res, ticks, time.time() - t0


def test_warm_model_background_returns_immediately_and_does_not_block_loop(monkeypatch):
    import asyncio
    s = FakeOllama(slow=2.0)                                                    # a 2 s "cold load" on /api/generate
    monkeypatch.setattr(llm, "API_BASE", s.url + "/v1"); monkeypatch.setattr(llm, "OLLAMA_API", None)
    monkeypatch.setattr(llm, "KEEP_ALIVE", "-1")
    async def go():
        t0 = time.time(); task = llm.warm_model_background("llama3.2:3b"); spawn = time.time() - t0
        _, ticks, secs = await _ticks_during(asyncio.wait_for(asyncio.shield(task), 10))
        return spawn, ticks, secs, task.result()
    try:
        spawn, ticks, secs, ok = asyncio.run(go())
    finally:
        s.close()
    assert spawn < 0.05 and ok is True                                          # handler-side cost is ~0
    assert secs >= 1.9 and ticks >= 40, f"event loop starved during warm: {ticks} ticks in {secs:.1f}s"
    monkeypatch.setattr(llm, "KEEP_ALIVE", None); assert llm.warm_model_background("x") is None


def test_init_llm_warms_off_the_event_loop(monkeypatch):
    import asyncio
    s = FakeOllama(slow=1.5)
    monkeypatch.setattr(llm, "API_BASE", s.url + "/v1"); monkeypatch.setattr(llm, "OLLAMA_API", None)
    monkeypatch.setattr(llm, "KEEP_ALIVE", "-1")
    async def ok(): return True
    monkeypatch.setattr(llm, "test_model", ok)
    try:
        _, ticks, secs = asyncio.run(_ticks_during(llm.init_llm()))
    finally:
        s.close()
    assert secs >= 1.4 and ticks >= 30, f"init_llm blocked the loop: {ticks} ticks in {secs:.1f}s"


# ── integration / functional (HTTP + socket.io handlers) ────────────────────
# app.api.routes hard-imports the RAG/document/image stack; skip (not fail) on a minimal chatbot install.
def _routes():
    for dep in ("pandas", "pypandoc", "pypdf", "weaviate", "bs4", "aiohttp", "PIL", "pillow_heif", "slowapi", "socketio"):
        pytest.importorskip(dep)
    from app.api import routes
    return routes


def test_models_detail_route_and_model_list_command(ollama, monkeypatch):
    routes = _routes()
    from fastapi.testclient import TestClient
    monkeypatch.setattr(routes, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    with TestClient(routes.app) as c:
        assert c.get("/models").json() == ["llama3:8b", "llama3.2:3b"]
        d = c.get("/models/detail").json()
        assert d[0]["id"] == "llama3:8b" and d[0]["resident"] is True and d[1]["resident"] is False
    html = (CHATBOT / "app/templates/index.html").read_text()
    assert "/models/detail" in html and "●" in html


def test_model_switch_and_detail_route_do_not_block_event_loop(monkeypatch):
    import asyncio, httpx
    routes = _routes()
    s = FakeOllama(models=["llama3:8b", "llama3.2:3b"], slow=2.0, slow_get=1.0)   # cold load 2 s, /api/ps 1 s
    monkeypatch.setattr(llm, "API_BASE", s.url + "/v1"); monkeypatch.setattr(llm, "OLLAMA_API", None)
    monkeypatch.setattr(llm, "KEEP_ALIVE", "-1"); llm._RESIDENT_CACHE.update(ts=0.0, models=None)
    monkeypatch.setattr(routes, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    emitted = []
    async def emit(*a, **k): emitted.append(a)
    monkeypatch.setattr(routes.sio, "emit", emit)
    monkeypatch.setitem(routes.client, "sid1", {"model": "llama3:8b", "context": []})
    async def go():
        t0 = time.time(); await routes.change_model("sid1", "llama3.2:3b"); switch = time.time() - t0
        t0 = time.time(); await routes.handle_model_command("sid1", "/model llama3:8b"); cmd = time.time() - t0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=routes.app), base_url="http://t") as c:
            r, ticks, secs = await _ticks_during(c.get("/models/detail"))
        await asyncio.gather(*list(llm._WARM_TASKS))                           # let the background pins finish
        return switch, cmd, r, ticks, secs
    try:
        switch, cmd, r, ticks, secs = asyncio.run(go())
    finally:
        s.close()
    assert switch < 0.5 and cmd < 0.5, f"model switch waited on the cold load ({switch:.2f}s / {cmd:.2f}s)"
    assert routes.client["sid1"]["model"] == "llama3:8b"
    assert r.status_code == 200 and secs >= 0.9 and ticks >= 20, f"/models/detail blocked the loop: {ticks} ticks in {secs:.1f}s"
    assert sum(1 for c in s.calls if c[0] == "POST" and c[1] == "/api/generate") == 2   # both switches were pinned


# ── frame ───────────────────────────────────────────────────────────────────
def test_modules_import_and_env_documented():
    import app.core.config as cfg
    assert hasattr(cfg, "KEEP_ALIVE") and hasattr(cfg, "OLLAMA_API")
    readme = (CHATBOT / "README.md").read_text()
    assert "KEEP_ALIVE" in readme and "OLLAMA_API" in readme
