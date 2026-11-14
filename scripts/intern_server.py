"""Intern-Decision (4B or 2B) behind the same ``/v1/systemone`` shape the other decision servers speak (standard library HTTP).

The model ships only an in-process ``DecisionEngine`` (``inference.py`` beside its weights); this wraps it so the
benchmark's ``DeciderRung`` can call it like Ollama or the Decider server. Run it in its own venv (its pins differ
from this project's: torch 2.9.1, transformers 5.14.1; plus causal-conv1d, flash-linear-attention and triton-windows for
Qwen3.5's fast path, built 2026-10-18 against D:\\Toolkits\\cuda-12.8), on native Windows:

    D:\\Models\\decision-servers\\intern\\.venv\\Scripts\\python.exe scripts\\intern_server.py --model <snapshot dir> --port 8767

- ``GET /health`` -> ``{"status": "ok"}`` once the model is loaded.
- ``POST /v1/systemone`` -> the engine's reply. Images arrive as base64 (raw or data URL) and are written to a temp
  file, because the engine reads local paths.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def patch_embed_as_linear(model) -> int:
    """Run the vision patch embedding as a matmul instead of its Conv3d (same math: kernel == stride, one output per patch).

    torch 2.9.1 with its bundled cuDNN 9.10 sends this bf16 Conv3d to ``slow_conv_dilated3d``, about 550,000 tiny copies:
    ~8.7 s of a ~8.8 s forward pass on the 3090 (profiled 2026-10-18). Returns how many modules were patched.
    """
    n = 0
    for module in model.modules():
        if type(module).__name__.endswith("VisionPatchEmbed"):
            proj = module.proj
            weight = proj.weight.reshape(proj.out_channels, -1)

            def forward(hidden_states, weight=weight, bias=proj.bias):
                return (hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype) @ weight.T) + bias

            module.forward = forward
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the Intern-Decision-4B snapshot folder (holds inference.py)")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.model)))
    from inference import DecisionEngine  # the model's own module, from its folder

    engine = DecisionEngine(checkpoint=args.model, device=args.device)
    print(f"patch embed as linear: {patch_embed_as_linear(engine.backend.model)} module(s)", flush=True)
    lock = threading.Lock()
    tmp = Path(tempfile.mkdtemp(prefix="intern-frames-"))
    model_dir = Path(args.model)  # a hub snapshot (models--internlm--Intern-Decision-2B/snapshots/<hash>) or a plain folder
    name = model_dir.parent.parent.name.split("--")[-1] if model_dir.parent.name == "snapshots" else model_dir.name

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, {"status": "ok", "model": name} if self.path == "/health" else {"detail": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(404, {"detail": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                paths = []
                for i, img in enumerate(body.pop("images", None) or []):
                    raw = img.split(",", 1)[1] if img.startswith("data:") else img
                    p = tmp / f"{threading.get_ident()}-{i}.png"
                    p.write_bytes(base64.b64decode(raw))
                    paths.append(str(p))
                body.pop("model", None)
                if paths:
                    body["images"] = paths
                with lock:
                    reply = engine.predict(body)
                self._send(200, reply)
            except Exception as e:  # noqa: BLE001
                self._send(422, {"detail": f"{type(e).__name__}: {e}"})

        def log_message(self, *a: object) -> None:  # quiet
            pass

    print(f"{name} on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
