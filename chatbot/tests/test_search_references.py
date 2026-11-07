"""Web search (SearXNG) query encoding and HTML escaping of search/RAG references.
References are emitted with voice 'ref' and inserted into the page as HTML.
Run: pytest chatbot/tests -q"""
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


class FakeSearx:
    """Minimal SearXNG: records the raw query string, returns canned results."""
    def __init__(self, results):
        self.paths = []
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                srv.paths.append(self.path)
                data = json.dumps({"results": results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def routes(monkeypatch):
    r = _routes()
    monkeypatch.setitem(r.client, "sid1", {"references": ""})
    return r


def test_search_query_is_url_encoded(routes, monkeypatch):
    s = FakeSearx([{"title": "t", "url": "https://a.example", "content": "c"}])
    monkeypatch.setattr(routes, "SEARXNG", s.url)
    try:
        for query in ("AT&T stock price", "C# vs C++ #tips", "what is 1+1?"):
            asyncio.run(routes.search_web("sid1", query, 5))
            qs = parse_qs(urlparse(s.paths[-1]).query)
            assert qs["q"] == [query]
            assert qs["format"] == ["json"]
    finally:
        s.close()


def test_search_references_are_html_escaped(routes, monkeypatch):
    s = FakeSearx([
        {"title": "<img src=x onerror=alert(1)>", "url": "https://ok.example/?a=1&b=2", "content": "x"},
        {"title": "quote", "url": "https://evil.example/' onmouseover='alert(1)", "content": "x"},
        {"title": "script", "url": "javascript:alert(1)", "content": "x"},
    ])
    monkeypatch.setattr(routes, "SEARXNG", s.url)
    try:
        asyncio.run(routes.search_web("sid1", "anything", 5))
    finally:
        s.close()
    refs = routes.client["sid1"]["references"]
    assert "<img" not in refs and "&lt;img src=x onerror=alert(1)&gt;" in refs
    assert "href='https://ok.example/?a=1&amp;b=2'" in refs
    assert "' onmouseover='" not in refs
    assert "javascript:" not in refs


def test_rag_references_are_html_escaped(routes, monkeypatch):
    from app.rag import rag
    docs = [{"title": "<script>alert(1)</script>", "file": "<b>f.pdf</b>", "content": "body", "chunk": ""}]
    monkeypatch.setattr(rag.rag_documents, "get_documents", lambda *a, **k: docs)
    content, references = rag.query_index("q", "lib", 1)
    assert "<script>" not in references and "&lt;script&gt;" in references
    assert "&lt;b&gt;f.pdf&lt;/b&gt;" in references
    assert "<script>alert(1)</script>" in content                     # prompt text to the LLM is unchanged
