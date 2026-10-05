"""Fake Ollama server for router tests: /api/tags, /api/ps, /api/generate, /api/embed, /v1/models,
/v1/chat/completions (plain + stream). Runs in a thread; nothing touches a real model."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeOllama:
    def __init__(self, models=("llama3:8b", "llama3.2:3b"), loaded=("llama3:8b",), fail=False, slow=0.0, reply="pong"):
        self.models, self.loaded, self.fail, self.slow, self.reply = list(models), set(loaded), fail, slow, reply
        self.calls = []
        self.lock = threading.Lock()
        srv = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, obj, ctype="application/json"):
                data = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                with srv.lock: srv.calls.append(("GET", self.path))
                if srv.fail: return self._send(500, {"error": "down"})
                if self.path == "/api/tags": return self._send(200, {"models": [{"name": m} for m in srv.models]})
                if self.path == "/api/ps": return self._send(200, {"models": [{"name": m, "expires_at": "2319-01-01T00:00:00Z"} for m in srv.loaded]})
                if self.path == "/v1/models": return self._send(200, {"object": "list", "data": [{"id": m} for m in srv.models]})
                self._send(404, {"error": "nope"})
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
                with srv.lock: srv.calls.append(("POST", self.path, body))
                if srv.fail: return self._send(500, {"error": "down"})
                time.sleep(srv.slow)
                if self.path in ("/api/generate", "/api/embed"):
                    srv.loaded.add(body.get("model")); return self._send(200, {"done": True})
                if self.path == "/v1/chat/completions":
                    srv.loaded.add(body.get("model"))
                    if body.get("stream"):
                        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                        for tok in (srv.reply, "!"):
                            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n"); return
                    return self._send(200, {"id": "x", "object": "chat.completion", "model": body.get("model"),
                                            "choices": [{"index": 0, "message": {"role": "assistant", "content": srv.reply}, "finish_reason": "stop"}]})
                self._send(404, {"error": "nope"})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self): self.httpd.shutdown()


@pytest.fixture
def fake_ollama():
    servers = []
    def make(**kw):
        s = FakeOllama(**kw); servers.append(s); return s
    yield make
    for s in servers: s.close()
