"""Every decision as one JSON line, with its camera frame saved beside it.

The log is the project's core asset: the benchmark replays it through each model, stage 3
fine-tunes on it (labelled by the sim's truth), and the charts come from it. A decision is
written when its truth is known, which for most questions is right away (the sim knows) and
for "in_bin" is after the part has settled.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .decider import Decision


def encode_png_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@dataclass
class Logbook:
    directory: Path
    episode: str = "episode"
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        (self.directory / "frames").mkdir(parents=True, exist_ok=True)
        self._file = (self.directory / "decisions.jsonl").open("a", encoding="utf-8")

    def save_frame(self, rgb: np.ndarray, tag: str) -> str:
        from PIL import Image

        name = f"{self.episode}_{self._count:03d}_{tag}.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(self.directory / "frames" / name)
        return f"frames/{name}"

    def record(self, decision: Decision, **extra: Any) -> dict[str, Any]:
        row = {"episode": self.episode, "index": self._count, **decision.as_dict(), **extra}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        self._count += 1
        return row

    def close(self) -> None:
        self._file.close()


def read_decisions(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
