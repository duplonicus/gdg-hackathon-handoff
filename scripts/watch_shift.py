"""Film one capsule shift from the cell's overview camera, a frame every --every s of sim, plus a contact sheet (Nov 6).

The critique rule: before a viewer run goes to dup, look at the carries, grabs, drops and anything showing through the
floor myself. Runs one shift as run_capsules.py does (lean, oracle) and saves `frames/tNNNN.png` and `sheet.png`.

    .\\run.ps1 scripts/watch_shift.py --seed 5000 --out output/capsules/watch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode  # noqa: E402
from handoff.decider import Ladder, Step  # noqa: E402
from handoff.logbook import Logbook  # noqa: E402
from handoff.rungs import OracleRung  # noqa: E402
from handoff.skills import TRAVEL_POSE  # noqa: E402
from handoff.tasks.capsules import CapsuleContext, run_capsules  # noqa: E402

LEAN = {"sensors": None, "help_text": False, "line_step_m": 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--every", type=float, default=1.0, help="sim seconds between frames")
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "capsules" / "watch"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    from dataclasses import replace

    from handoff.sim.base import CameraSpec

    episode = make_capsule_episode(cells, args.seed)
    cell = cells.cell(episode.work_cell or "a")
    lay = cells.layout
    look = (*cell.to_world(0.32, 0.0), lay.table_top + 0.05)
    # The overview camera sits behind the robot, whose back hides the hands: two side views across the table instead.
    side_cams = tuple(CameraSpec(f"watch_{k}", pos=(*cell.to_world(0.55, s * 0.95), lay.table_top + 0.55), lookat=look, fov=50, res=(640, 400))
                      for k, s in (("left", 1.0), ("right", -1.0)))
    scene = make_two_cell_scene(cells)
    sim = GenesisSimulator(args.config, **LEAN)
    sim.build(replace(scene, cameras=scene.cameras + side_cams))
    shots: list[Path] = []
    next_t = [0.0]
    real_step = sim.step

    def step(n: int = 1) -> None:
        real_step(n)
        if sim.time >= next_t[0]:
            path = out / "frames" / f"t{len(shots):04d}.png"
            Image.fromarray(np.concatenate([sim.capture("watch_left").rgb, sim.capture("watch_right").rgb], axis=1)).save(path)
            shots.append(path)
            next_t[0] = sim.time + args.every

    sim.step = step  # type: ignore[method-assign]
    place_capsule_episode(sim, cells, episode)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    holder: dict = {}
    ctx = CapsuleContext(cells=cells, episode=episode, block=16, logbook=Logbook(out),
                         ladder=Ladder([Step(OracleRung(lambda q: holder["ctx"].truth_now), 0.5)]),
                         on_event=lambda e: print(f"  {e}", flush=True) if not e.startswith("asked_") else None)
    holder["ctx"] = ctx
    result = run_capsules(sim, ctx)
    print(json.dumps({"seed": args.seed, "status": result.status.value, "checks": result.checks, "frames": len(shots)}), flush=True)
    # contact sheet: every frame, 8 across, at a quarter size
    tiles = [Image.open(p).reduce(3) for p in shots]
    if tiles:
        w, h = tiles[0].size
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, h * rows), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * w, (i // cols) * h))
        sheet.save(out / "sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
