"""Chatbot resident-model awareness — categories: unit, security, performance, retry, integration, functional, frame.
Run: pytest chatbot/tests -q   (the fake Ollama fixture lives in router/tests/conftest.py and is reused here)."""
import os, subprocess, sys, time
from pathlib import Path

import pytest

CHATBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT)); sys.path.insert(0, str(CHATBOT.parent / "router" / "tests"))
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


# ── integration / functional (HTTP routes) ──────────────────────────────────
def test_models_detail_route_and_model_list_command(ollama, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import routes
    monkeypatch.setattr(routes, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    monkeypatch.setattr(llm, "get_models", lambda: ["llama3:8b", "llama3.2:3b"])
    with TestClient(routes.app) as c:
        assert c.get("/models").json() == ["llama3:8b", "llama3.2:3b"]
        d = c.get("/models/detail").json()
        assert d[0]["id"] == "llama3:8b" and d[0]["resident"] is True and d[1]["resident"] is False
    html = (CHATBOT / "app/templates/index.html").read_text()
    assert "/models/detail" in html and "●" in html


# ── frame ───────────────────────────────────────────────────────────────────
def test_modules_import_and_env_documented():
    import app.core.config as cfg
    assert hasattr(cfg, "KEEP_ALIVE") and hasattr(cfg, "OLLAMA_API")
    readme = (CHATBOT / "README.md").read_text()
    assert "KEEP_ALIVE" in readme and "OLLAMA_API" in readme
