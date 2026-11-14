"""Cutting logged tray pictures into blocks (Nov 4): every size covers every capsule, and the labels match the pictures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from handoff.capsule_cell import TwoCells

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("capsule_block_curve", ROOT / "scripts" / "capsule_block_curve.py")
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


def fake_log(tmp_path: Path, leaker_spots: set[int]) -> Path:
    """A one-scan log of cell a: a blank tray picture at 2 mm a pixel, every grid capsule with its box and truth."""
    from PIL import Image

    cells = TwoCells()
    lay, a = cells.layout, cells.cell("a")
    caps = []
    for i, (x, y) in enumerate(lay.grid()):
        wx, wy = a.to_world(x, y)
        r, c = int(round((wx + 2.4) * 500)), int(round((wy + 2.4) * 500))
        caps.append({"xyz": [wx, wy, 0.83], "box": [r - 6, c - 3, r + 6, c + 3], "leaker": i in leaker_spots, **({"pool_size": 1} if i in leaker_spots else {})})
    (tmp_path / "frames").mkdir()
    Image.fromarray(np.zeros((2400, 2400, 3), dtype=np.uint8)).save(tmp_path / "frames" / "tray.png")
    row = {"episode": "ep0000", "index": 1, "question": "leak_block", "truth": "no",
           "context": {"tray": {"capsules": caps, "cell": "a", "block": 16, "px_per_m": 500.0, "tray_frame": "frames/tray.png"}}}
    (tmp_path / "decisions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path


def test_every_size_covers_every_capsule_and_labels_what_it_shows(tmp_path):
    rows = len({round(x, 6) for x, _ in TwoCells().layout.grid()})
    spot = 2 * rows + 1  # column 2 of the right half: beside a 2 x 2 block's edge
    log = fake_log(tmp_path, {spot})
    n_caps = len(TwoCells().layout.grid())
    for size in (1, 4, 16, 64, 0):
        out = tmp_path / f"n{size}"
        curve.cut(log, size, out)
        rows = [json.loads(line) for line in (out / "decisions.jsonl").open(encoding="utf-8")]
        covered = sorted(i for r in rows for i in r["context"]["capsules"])
        assert covered == list(range(n_caps)), size  # each capsule in exactly one picture's own set
        for r in rows:
            assert (out / r["frame"]).exists()
            shows_leaker = spot in r["context"]["in_view"]
            if r["question"] == "leak_capsule":
                assert r["truth"] == ("leaker" if r["context"]["capsules"] == [spot] else "good")  # about the middle capsule only
            elif not shows_leaker:
                assert r["truth"] == "no"
            else:
                assert r["truth"] in ("yes", None)  # wholly in view, or only in part (left out of scoring)
                if spot in r["context"]["capsules"]:
                    assert r["truth"] == "yes"  # its own block's picture holds all of it
        if size == 0:
            assert len(rows) == 1 and rows[0]["truth"] == "yes"
        if size == 16:
            assert sum(1 for r in rows if r["truth"] == "yes") >= 1  # at least its own block
