"""Chatbot UI markdown rendering must not execute HTML/script from LLM output.
Runs the real marked.min.js + safe_markdown.js in V8 (py_mini_racer).
Run: pytest chatbot/tests -q"""
from pathlib import Path

import pytest

MiniRacer = pytest.importorskip("py_mini_racer").MiniRacer

CHATBOT = Path(__file__).resolve().parent.parent
STATIC = CHATBOT / "app" / "static"
INDEX = CHATBOT / "app" / "templates" / "index.html"


@pytest.fixture(scope="module")
def js():
    ctx = MiniRacer()
    ctx.eval("var window = this; var self = this;")
    ctx.eval((STATIC / "marked.min.js").read_text())
    ctx.eval((STATIC / "safe_markdown.js").read_text())
    return ctx


def render(ctx, text):
    return ctx.call("renderMarkdownSafe", text)


@pytest.mark.parametrize("payload", [
    "<img src=x onerror=alert(1)>",
    "hello <script>alert(1)</script>",
    "<div>\n<iframe src='https://evil.example'></iframe>\n</div>",
    "<svg onload=alert(1)>",
])
def test_raw_html_is_escaped(js, payload):
    out = render(js, payload)
    assert "<img" not in out and "<script" not in out and "<iframe" not in out and "<svg" not in out
    assert "&lt;" in out


@pytest.mark.parametrize("payload", [
    "[click](javascript:alert(1))",
    "[click](JavaScript:alert(1))",
    "[click](<java\tscript:alert(1)>)",
    "[click](vbscript:msgbox(1))",
    "[click](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "<javascript:alert(1)>",
    "![x](javascript:alert(1))",
])
def test_unsafe_link_schemes_are_neutralized(js, payload):
    out = render(js, payload).lower()
    for scheme in ("javascript:", "java\tscript:", "vbscript:", "data:text"):
        assert f'="{scheme}' not in out and f"='{scheme}" not in out
    assert 'href="#"' in out or 'src="#"' in out


def test_markdown_still_renders(js):
    out = render(js, "# Title\n\n**bold** and `x < y`\n\n[site](https://example.com)\n\n```python\nprint('<b>')\n```")
    assert "<h1>Title</h1>" in out
    assert "<strong>bold</strong>" in out
    assert "<code>x &lt; y</code>" in out
    assert '<a href="https://example.com">site</a>' in out
    assert "print(&#39;&lt;b&gt;&#39;)" in out


def test_safe_url_and_escape_helpers(js):
    assert js.call("safeUrl", "https://news.example/a?b=1") == "https://news.example/a?b=1"
    assert js.call("safeUrl", "javascript:alert(1)") == "#"
    assert js.call("escapeHtml", "x' onmouseover='alert(1)") == "x&#039; onmouseover=&#039;alert(1)"


def test_index_uses_safe_renderer():
    html = INDEX.read_text()
    assert "/static/safe_markdown.js" in html
    assert "marked.parse(" not in html                     # all markdown goes through renderMarkdownSafe
    assert ">${value}</textarea>" not in html              # settings values are escaped
    assert "\"<a href='\" + linkURL" not in html           # news links are escaped
