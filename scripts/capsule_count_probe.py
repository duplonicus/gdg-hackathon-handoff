"""How many loose capsules can Genesis carry on one tray before it slows to a crawl? (Nov 4 sizing probe.)

Builds the kit's robot (parked out of the way), a table, a tray with parchment, and N free capsules in a grid, with
a visual-only pool under every tenth one. Then it times physics steps and checks the capsules stayed put (a pile at
rest must not jitter, sink or explode). One N per process: Genesis builds a scene once per process.

    .\\run.ps1 scripts/capsule_count_probe.py --n 140 --out output/bench/capsule_count.jsonl --frame output/bench/capsule_140.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.sim.base import BoxSpec, CameraSpec, SceneSpec  # noqa: E402

TABLE_TOP = 0.82
TRAY = (0.45, 0.35)  # a cafeteria tray, m (x, y)
TRAY_CENTER = (0.60, 0.30)
CAPSULE = (0.024, 0.011, 0.011)  # length, width, height: a round soft gel (--height below 0.011 flattens it)
PITCH = (0.030, 0.016)  # grid spacing along x and y
PAPER_TOP = TABLE_TOP + 0.011  # the parchment is 1 mm thick on a 1 cm tray floor


def grid(n: int) -> list[tuple[float, float]]:
    """The first n spots of the tray's capsule grid, row by row, centred on the tray."""
    nx = int((TRAY[0] - 0.03) // PITCH[0])
    ny = int((TRAY[1] - 0.03) // PITCH[1])
    if n > nx * ny:
        raise SystemExit(f"--n {n} does not fit: the grid holds {nx * ny}")
    x0 = TRAY_CENTER[0] - (nx - 1) * PITCH[0] / 2.0
    y0 = TRAY_CENTER[1] - (ny - 1) * PITCH[1] / 2.0
    return [(x0 + (i % nx) * PITCH[0], y0 + (i // nx) * PITCH[1]) for i in range(n)]


def scene(n: int, free: int | None = None, capsule: tuple[float, float, float] = CAPSULE) -> SceneSpec:
    """``free``: only the first ``free`` capsules are loose bodies, the rest fixed in place (None: all loose)."""
    cx, cy = TRAY_CENTER
    tray_z = TABLE_TOP + 0.005
    paper_top = PAPER_TOP
    boxes = [
        BoxSpec("table", (0.80, 1.20, TABLE_TOP), (0.65, 0.20, TABLE_TOP / 2.0), color=(0.55, 0.40, 0.25), fixed=True),
        BoxSpec("tray_floor", (*TRAY, 0.010), (cx, cy, tray_z), color=(0.62, 0.64, 0.66), fixed=True),
        BoxSpec("parchment", (TRAY[0] - 0.02, TRAY[1] - 0.02, 0.001), (cx, cy, PAPER_TOP - 0.0005), color=(0.93, 0.89, 0.80), fixed=True, look="matte"),
    ]
    for i, (x, y) in enumerate(grid(n)):
        z = paper_top + capsule[2] / 2.0
        boxes.append(BoxSpec(f"capsule_{i:03d}", capsule, (x, y, z), color=(0.85, 0.62, 0.15), rho=1100.0, shape="capsule", look="glossy", fixed=free is not None and i >= free))
        if i % 10 == 0:
            boxes.append(
                BoxSpec(f"pool_{i:03d}", (0.030, 0.030, 0.0008), (x + 0.004, y, paper_top + 0.0004), color=(0.80, 0.58, 0.12), fixed=True, shape="disc", look="glossy", collision=False)
            )
    cams = (CameraSpec("tray", pos=(cx - 0.25, cy - 0.05, TABLE_TOP + 0.40), lookat=(cx, cy, TABLE_TOP), fov=50, res=(960, 600)),)
    return SceneSpec(boxes=tuple(boxes), robot_start=(-1.2, -1.0, 0.0), cameras=cams)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--free", type=int, default=None, help="only this many capsules loose, the rest fixed")
    parser.add_argument("--height", type=float, default=CAPSULE[2], help="capsule height, m (its width, 0.011, is a round pill)")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "bench" / "capsule_count.jsonl"))
    parser.add_argument("--frame", default="", help="save the tray camera's view here (PNG)")
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator

    spec = scene(args.n, args.free, (CAPSULE[0], CAPSULE[1], args.height))
    names = [b.name for b in spec.boxes if b.shape == "capsule" and not b.fixed]
    sim = GenesisSimulator(args.config)
    began = time.perf_counter()
    sim.build(spec)
    build_s = time.perf_counter() - began
    start = {name: np.array(sim.object_state(name).pos) for name in names}
    sim.step(50)  # warm-up: the first steps compile kernels
    began = time.perf_counter()
    sim.step(args.steps)
    step_s = time.perf_counter() - began
    moved = [float(np.linalg.norm(np.array(sim.object_state(name).pos) - start[name])) for name in names]
    zs = [float(sim.object_state(name).pos[2]) for name in names]
    row = {
        "n": args.n,
        "free": len(names),
        "bodies": len(spec.boxes),
        "build_s": round(build_s, 1),
        "steps_per_s": round(args.steps / step_s, 1),
        "real_time_x": round(args.steps * sim.dt / step_s, 2),
        "max_moved_mm": round(1000 * max(moved), 2) if moved else 0.0,
        "median_moved_mm": round(1000 * float(np.median(moved)), 2) if moved else 0.0,
        "height_mm": round(1000 * args.height, 2),
        "lowest_centre_mm_above_paper": round(1000 * (min(zs) - PAPER_TOP), 2) if zs else None,  # height / 2 when resting
        "dt": sim.dt,
        "steps": args.steps,
    }
    if args.frame:
        from PIL import Image

        rgb = sim.capture("tray").rgb
        Path(args.frame).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.frame)
        row["frame"] = args.frame
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
