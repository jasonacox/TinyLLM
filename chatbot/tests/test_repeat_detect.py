"""/repeat on|off must control repetition detection in the streaming loop for that session.
Drives the real send_update() task with a fake token stream; no LLM involved.
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


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _run_stream(routes, monkeypatch, repeat_detect):
    """Connect a session, stream a highly repetitive answer, return the emitted updates."""
    emitted = []

    async def emit(event, data=None, **kwargs):
        emitted.append(data or {})

    # 3,000 chars of the same 10-char phrase: well past REPEAT_WINDOW * (REPEAT_COUNT + 1)
    tokens = ["la la la. "] * 300

    async def fake_ask(prompt, sid, sio):
        return iter([_chunk(t) for t in tokens])

    monkeypatch.setattr(routes.sio, "emit", emit)
    monkeypatch.setattr(routes, "ask", fake_ask)
    sid = f"repeat-{repeat_detect}"

    async def go():
        await routes.handle_connect(sid, {})
        routes.client[sid]["cot"] = False
        routes.client[sid]["think"] = False
        routes.client[sid]["repeat_detect"] = repeat_detect
        routes.client[sid]["prompt"] = "sing"
        for _ in range(100):
            await asyncio.sleep(0.05)
            if any(d.get("voice") == "done" for d in emitted):
                break
        routes.client[sid]["stop_thread_flag"] = True
        await asyncio.sleep(0.15)
        routes.client.pop(sid, None)

    asyncio.run(go())
    return emitted


def test_repeat_detection_on_stops_stream(monkeypatch):
    routes = _routes()
    emitted = _run_stream(routes, monkeypatch, repeat_detect=True)
    assert any("Repeating Text Detected" in str(d.get("update")) for d in emitted)


def test_repeat_off_streams_full_answer(monkeypatch):
    routes = _routes()
    emitted = _run_stream(routes, monkeypatch, repeat_detect=False)
    assert not any("Repeating Text Detected" in str(d.get("update")) for d in emitted)
    streamed = "".join(d["update"] for d in emitted if d.get("voice") == "ai")
    assert len(streamed) == 3000
