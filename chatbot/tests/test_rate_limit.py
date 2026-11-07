"""LLM provider rate limits (HTTP 429, issue #17): tell the user when to retry, keep the
conversation, and keep the session's streaming task alive for the next prompt.
Run: pytest chatbot/tests -q"""
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

CHATBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT))
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")      # never a real server during import
os.environ.setdefault("PROMPT_FILE", str(CHATBOT / ".pytest-prompts.json"))
from app.core import llm   # noqa: E402

AZURE_429 = ("Error code: 429 - Requests to the ChatCompletions_Create Operation under Azure OpenAI API "
             "version 2023-05-15 have exceeded token rate limit. Please retry after 86400 seconds.")


def _rate_limit(msg=AZURE_429, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    req = httpx.Request("POST", "http://llm/v1/chat/completions")
    return openai.RateLimitError(msg, response=httpx.Response(429, request=req, headers=headers), body=None)


@pytest.mark.parametrize("err,expected", [
    (_rate_limit(), "about 24.0 hours"),
    (_rate_limit("slow down", retry_after=20), "about 20 seconds"),
    (_rate_limit("slow down", retry_after=600), "about 10 minutes"),
    (_rate_limit("slow down"), "please wait a moment"),
])
def test_rate_limit_message(err, expected):
    msg = llm.rate_limit_message(err)
    assert msg.startswith("[LLM rate limit reached") and expected in msg


def _routes():
    for dep in ("pandas", "pypandoc", "pypdf", "weaviate", "bs4", "aiohttp", "PIL", "pillow_heif", "slowapi", "socketio"):
        pytest.importorskip(dep)
    from app.api import routes
    return routes


class FlakyLLM:
    """First request is rate limited, later ones stream 'hello'."""
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _rate_limit(retry_after=30)
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
        return iter([chunk])


def test_rate_limited_prompt_keeps_session_and_conversation(monkeypatch):
    routes = _routes()
    emitted = []

    async def emit(event, data=None, **kwargs):
        emitted.append(data or {})

    monkeypatch.setattr(routes.sio, "emit", emit)
    monkeypatch.setattr(llm, "llm_stream", FlakyLLM())
    sid = "rate-limit-sid"

    async def wait_done(n):
        for _ in range(100):
            await asyncio.sleep(0.05)
            if sum(1 for d in emitted if d.get("voice") == "done") >= n:
                return

    async def go():
        await routes.handle_connect(sid, {})
        s = routes.client[sid]
        s.update(cot=False, think=False)
        history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
        s["context"] = s["context"][:1] + history
        s["prompt"] = "first"
        await wait_done(1)
        context_after_429 = list(s["context"])
        s["prompt"] = "second"                  # the session task must still be running
        await wait_done(2)
        s["stop_thread_flag"] = True
        await asyncio.sleep(0.15)
        routes.client.pop(sid, None)
        return context_after_429

    context_after_429 = asyncio.run(go())
    updates = [str(d.get("update")) for d in emitted]
    assert any("rate limit reached - please try again in about 30 seconds" in u for u in updates)
    assert not any("An error occurred" in u for u in updates)
    assert context_after_429[1:] == [{"role": "user", "content": "earlier question"},
                                     {"role": "assistant", "content": "earlier answer"}]
    assert any(d.get("voice") == "ai" and d.get("update") == "hello" for d in emitted)
