"""Optional: push the harness snapshot to a remote viewer (the Vercel app in web/) every few seconds.

The engine stays in this process; the viewer only displays what is published. Standard library only.
  PAPER_PUBLISH_URL    e.g. https://<project>.vercel.app/api/ingest
  PAPER_PUBLISH_TOKEN  the ingest token (environment only: it is never a command-line argument and never part of the published data)
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from paper.server import jsonable

MAX_BYTES = 3_500_000                       # the viewer's function accepts at most ~4.5 MB


class Publisher(threading.Thread):
    def __init__(self, runner, url: str, token: str, every: float = 15.0, timeout: float = 10.0):
        super().__init__(daemon=True)
        if not url.startswith("https://") and not url.startswith("http://localhost") and not url.startswith("http://127.0.0.1"):
            raise ValueError("the publish URL must be https (http only for localhost)")
        self.runner, self.url, self._token, self.every, self.timeout = runner, url, token, every, timeout
        self.stop_flag = threading.Event()
        self.ok = self.fail = 0
        self.last_error = None

    def once(self) -> bool:
        body = json.dumps(jsonable(self.runner.snapshot()), separators=(",", ":")).encode()
        if len(body) > MAX_BYTES:
            self.last_error = f"snapshot too large ({len(body)} bytes)"
            self.fail += 1
            return False
        req = urllib.request.Request(self.url, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._token, "User-Agent": "topstep-paper-publisher"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                ok = 200 <= r.status < 300
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code}" + (" (token rejected)" if e.code in (401, 403) else "")
            ok = False
        except Exception as e:                                  # network down, DNS, timeout: keep going
            self.last_error = type(e).__name__
            ok = False
        if ok:
            self.ok += 1
            self.last_error = None
        else:
            self.fail += 1
        return ok

    def run(self) -> None:
        streak = 0
        while not self.stop_flag.is_set():
            if self.once():
                streak = 0
            else:
                streak += 1
                if streak in (1, 20) or streak % 240 == 0:
                    print(f"publish failed ({self.last_error}); {streak} in a row, still trying", flush=True)
            self.stop_flag.wait(self.every)

    def stop(self) -> None:
        self.stop_flag.set()


def from_env(runner, every: float = 15.0):
    url, token = os.environ.get("PAPER_PUBLISH_URL"), os.environ.get("PAPER_PUBLISH_TOKEN")
    if not url and not token:
        return None
    if not (url and token):
        raise SystemExit("set both PAPER_PUBLISH_URL and PAPER_PUBLISH_TOKEN (or neither)")
    return Publisher(runner, url, token, every)
