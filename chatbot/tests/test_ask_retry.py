"""ask() retries after a context-length / missing-model error must not resend the prompt twice.
Uses a fake OpenAI client; nothing touches a real model.
Run: pytest chatbot/tests -q"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import openai
import pytest

CHATBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHATBOT))
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:1/v1")      # never a real server during import
os.environ.setdefault("PROMPT_FILE", str(CHATBOT / ".pytest-prompts.json"))
from app.core import llm   # noqa: E402


def _bad_request(msg):
    req = httpx.Request("POST", "http://llm/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


class FakeCompletions:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append([dict(m) for m in kwargs["messages"]])
        if self.errors:
            raise _bad_request(self.errors.pop(0))
        return iter(())


class FakeStream:
    def __init__(self, errors):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(errors)


class FakeSio:
    async def emit(self, *args, **kwargs):
        pass


@pytest.fixture
def session(monkeypatch):
    async def no_sleep(*_):
        pass
    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)
    sid = "retry-sid"
    history = [{"role": "system", "content": "sys"}]
    for i in range(4):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    monkeypatch.setitem(llm.client, sid, {"context": history, "image_data": "", "model": "m", "remember": True})
    return sid


def _user_turns(messages, text):
    return sum(1 for m in messages if m["role"] == "user" and text in m["content"])


def test_context_length_retry_sends_prompt_once(monkeypatch, session):
    fake = FakeStream(["This model's maximum context length is 4096 tokens"])
    monkeypatch.setattr(llm, "llm_stream", fake)
    asyncio.run(llm.ask("short question", session, FakeSio()))
    first, retry = fake.chat.completions.calls
    assert _user_turns(retry, "short question") == 1
    assert len(retry) < len(first), "retry should send a smaller context, not a larger one"


def test_long_prompt_is_replaced_not_duplicated(monkeypatch, session):
    fake = FakeStream(["maximum context length exceeded"])
    monkeypatch.setattr(llm, "llm_stream", fake)
    long_prompt = "START " + "x" * 4000 + " END"
    asyncio.run(llm.ask(long_prompt, session, FakeSio()))
    retry = fake.chat.completions.calls[-1]
    assert _user_turns(retry, "START") == 1
    assert len(retry[-1]["content"]) < len(long_prompt)


def test_context_after_retry_has_single_prompt(monkeypatch, session):
    fake = FakeStream(["maximum context length", "maximum context length"])
    monkeypatch.setattr(llm, "llm_stream", fake)
    asyncio.run(llm.ask("hello again", session, FakeSio()))
    assert _user_turns(llm.client[session]["context"], "hello again") == 1
