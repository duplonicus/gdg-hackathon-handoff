"""pplx-decider-flash-v1.2 behind the same ``/v1/systemone`` shape the other decision servers speak (standard library HTTP).

The model ships an in-process ``Decider`` (``source/src/pplx_decider`` beside its weights); this wraps it so the
benchmark's ``DeciderRung`` can call it like the Intern and d3 servers. Run it in its own venv (the model's pins:
torch 2.14.0, transformers 5.17.0; ``D:\\Models\\decision-servers\\pplx\\setup.ps1``), on native Windows:

    D:\\Models\\decision-servers\\pplx\\.venv\\Scripts\\python.exe scripts\\pplx_server.py --model <snapshot dir> --port 8769

- ``GET /health`` -> ``{"status": "ok"}`` once the model is loaded.
- ``POST /v1/systemone`` -> the decider's reply. Images arrive as data URLs, which ``decide`` reads directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the pplx-decider-flash-v1.2 snapshot folder (holds source/src)")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.model) / "source" / "src"))
    from pplx_decider import Decider  # the model's own package, from its folder

    decider = Decider.from_pretrained(Path(args.model), device=args.device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(
                200,
                {"status": "ok", "model": "pplx-decider-flash-v1.2"}
                if self.path == "/health"
                else {"detail": "not found"},
            )

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(404, {"detail": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                body.pop("model", None)
                with lock:
                    reply = decider.decide(body)
                self._send(200, reply)
            except Exception as e:  # noqa: BLE001
                self._send(422, {"detail": f"{type(e).__name__}: {e}"})

        def log_message(self, *a: object) -> None:  # quiet
            pass

    print(f"pplx-decider-flash-v1.2 on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
