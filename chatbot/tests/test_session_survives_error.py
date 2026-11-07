"""A failed LLM request must not kill the session's streaming task (later prompts must still work).
Run: pytest chatbot/tests -q"""
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CHATBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT))
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")      # never a real server during import
os.environ.setdefault("PROMPT_FILE", str(CHATBOT / ".pytest-prompts.json"))


def _routes():
    for dep in ("pandas", "pypandoc", "pypdf", "weaviate", "bs4", "aiohttp", "PIL", "pillow_heif", "slowapi", "socketio"):
        pytest.importorskip(dep)
    from app.api import routes
    return routes


def test_first_prompt_error_does_not_kill_session(monkeypatch):
    routes = _routes()
    emitted = []
    calls = {"n": 0}

    async def emit(event, data=None, **kwargs):
        emitted.append(data or {})

    async def flaky_ask(prompt, sid, sio):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("LLM unavailable")       # e.g. LLMError after retries are exhausted
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])])

    monkeypatch.setattr(routes.sio, "emit", emit)
    monkeypatch.setattr(routes, "ask", flaky_ask)
    sid = "survive-sid"

    async def go():
        await routes.handle_connect(sid, {})
        s = routes.client[sid]
        s.update(cot=False, think=False)
        s["prompt"] = "one"
        await asyncio.sleep(0.4)
        s["prompt"] = "two"
        await asyncio.sleep(0.4)
        pending = s["prompt"]
        s["stop_thread_flag"] = True
        await asyncio.sleep(0.15)
        routes.client.pop(sid, None)
        return pending

    pending = asyncio.run(go())
    assert any("An error occurred" in str(d.get("update")) for d in emitted)
    assert any(d.get("voice") == "ai" and d.get("update") == "hi" for d in emitted), "second prompt never answered"
    assert pending == ""
