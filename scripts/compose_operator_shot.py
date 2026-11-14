"""Lay the sim and the operator page side by side for the film (P1-B, P1-D in media/film/SCRIPT.md).

Inputs, all in one shot folder: the demo's film frames (<dir>/<angle>/fNNNNN.jpg and frames.jsonl, from
`demo.py --film`) and the redrawn page (page_shots/ and page_shots.jsonl, from render_operator_page.py). Each film
frame gets the page shot whose wall time is the latest at or before the frame's wall time, so the page changes on the
frame where it changed in the run. A question or a press that would flash by is held readable by repeating the
frame (the robot holds still while it waits). Writes <dir>/composed/fNNNNN.jpg (renumbered from 0) and
<dir>/composed_index.jsonl (each output frame's source frame and page state); join them with ffmpeg at the film's fps.

    .\\run.ps1 python scripts/compose_operator_shot.py output/film/p1b --angle front
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

W, H = 1920, 1080
SIM_W, SIM_H = 960, 540  # the sim on the left, the page (rendered 960 x 700) on the right at full size so its text reads, both centred
PAGE_W = W - SIM_W


def page_for(times: list[float], wall_t: float) -> int | None:
    """Index of the latest page shot at or before wall_t (None before the first)."""
    i = bisect.bisect_right(times, wall_t) - 1
    return i if i >= 0 else None


def plan(frames: list[dict], shots: list[dict], fps: float, hold_question: float, hold_press: float) -> list[tuple[int, int | None]]:
    """The output frames as (source frame, page shot index). The sim barely moves while the robot waits on the page,
    so a question would flash by in a few frames: each run of frames showing a question is held to at least
    ``hold_question`` seconds, and a press to ``hold_press``, by repeating the run's first frame (the robot is still)."""
    times = [s["wall_t"] for s in shots]
    tagged = [(f["frame"], page_for(times, f["wall_t"])) for f in frames]
    out: list[tuple[int, int | None]] = []
    i = 0
    while i < len(tagged):
        j = i
        while j < len(tagged) and tagged[j][1] == tagged[i][1]:
            j += 1
        k = tagged[i][1]
        state = shots[k]["state"] if k is not None else None
        need = round(fps * (hold_question if state == "question" else hold_press if state == "press" else 0))
        out += [tagged[i]] * max(0, need - (j - i))  # the freeze, before the run plays on
        out += tagged[i:j]
        i = j
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir")
    p.add_argument("--angle", default="front")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--hold-question", type=float, default=2.5, help="seconds a question stays readable at least")
    p.add_argument("--hold-press", type=float, default=0.8, help="seconds a pressed answer shows at least")
    a = p.parse_args()
    from PIL import Image

    d = Path(a.dir)
    frames = [json.loads(line) for line in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shots = [json.loads(line) for line in (d / "page_shots.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shots.sort(key=lambda s: s["wall_t"])
    out = d / "composed"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    cache: dict[int, Image.Image] = {}
    y = (H - SIM_H) // 2
    seq = plan(frames, shots, a.fps, a.hold_question, a.hold_press)
    index = []
    for n, (src, k) in enumerate(seq):
        canvas = Image.new("RGB", (W, H), (17, 17, 17))
        sim = Image.open(d / a.angle / f"f{src:05d}.jpg").resize((SIM_W, SIM_H), Image.LANCZOS)
        canvas.paste(sim, (0, y))
        if k is not None:
            if k not in cache:
                shot = Image.open(d / "page_shots" / shots[k]["shot"]).convert("RGB")
                scale = min(PAGE_W / shot.width, H / shot.height)  # the page may use the full height
                cache[k] = shot.resize((int(shot.width * scale), int(shot.height * scale)), Image.LANCZOS)
            img = cache[k]
            canvas.paste(img, (SIM_W + (PAGE_W - img.width) // 2, (H - img.height) // 2))
        canvas.save(out / f"f{n:05d}.jpg", quality=95)
        index.append({"out": n, "src": src, "page": shots[k]["state"] if k is not None else None})
    (d / "composed_index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n", encoding="utf-8")
    print(f"{len(seq)} composed frames ({len(seq) - len(frames)} held) in {out}; index in composed_index.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
