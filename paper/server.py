"""Tiny dependency-free web server for the paper-trading dashboard: GET / (the page), GET /api/state (JSON snapshot of the runner, refreshed by the page every 2 s)."""
from __future__ import annotations

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

STATIC = os.path.join(os.path.dirname(__file__), "static")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(float(x)) or math.isinf(float(x))) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def make_handler(runner):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):             # quiet
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "dashboard.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(jsonable(runner.snapshot())).encode(), "application/json")
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")
    return H


def serve(runner, host: str = "127.0.0.1", port: int = 8765, block: bool = True):
    srv = ThreadingHTTPServer((host, port), make_handler(runner))
    if block:
        srv.serve_forever()
    else:
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
    return srv
