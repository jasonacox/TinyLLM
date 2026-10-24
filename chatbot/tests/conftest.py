"""Fake Ollama server for chatbot tests: /api/ps, /api/generate, /v1/models.
Runs in a thread on loopback; nothing touches a real model."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, models=("llama3:8b", "llama3.2:3b"), loaded=("llama3:8b",), slow=0.0, slow_get=0.0):
        self.models = list(models)
        self.loaded = set(loaded)
        self.slow = slow                        # delay POSTs (a cold model load)
        self.slow_get = slow_get                # delay GETs (/api/ps)
        self.calls = []
        self.lock = threading.Lock()
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with srv.lock:
                    srv.calls.append(("GET", self.path))
                time.sleep(srv.slow_get)
                if self.path == "/api/ps":
                    return self._send(200, {"models": [{"name": m} for m in srv.loaded]})
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": m} for m in srv.models]})
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with srv.lock:
                    srv.calls.append(("POST", self.path, body))
                time.sleep(srv.slow)
                if self.path == "/api/generate":
                    srv.loaded.add(body.get("model"))
                    return self._send(200, {"done": True})
                return self._send(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
