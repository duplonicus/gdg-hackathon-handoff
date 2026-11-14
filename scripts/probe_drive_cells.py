"""Drive the MH3 between the capsule cells and under a table top, in Genesis (Nov 4).

From the spawn in the aisle: tuck in under cell a's table at the spots the left hand needs for a few tray points,
shift between them, back out, and park at cell b. Prints each move's error and time, and saves an overview frame at
every stop so the result can be looked at. Arms stay in the travel pose (forearms up) the whole time.

    .\\run.ps1 scripts/probe_drive_cells.py --out output/cells/drive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TwoCells, make_two_cell_scene, park_for  # noqa: E402
from handoff.drive import back_out, shift_under, tuck_in  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "drive"))
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator
    from handoff.skills import TRAVEL_POSE

    cells = TwoCells()
    spec = make_two_cell_scene(cells)
    sim = GenesisSimulator(args.config)
    sim.build(spec)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(1.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a, b = cells.cell("a"), cells.cell("b")
    lay = cells.layout
    grid = lay.grid()
    points = [("tray_near_left", grid[-10]), ("tray_far_right", grid[9]), ("tray_middle", grid[len(grid) // 2]), ("leaker_bin", lay.leaker_bin("left"))]
    rows = []

    def record(label: str, result, cam: str) -> None:
        pose = sim.base_pose()
        row = {"move": label, "ok": result.ok, "err_mm": round(1000 * result.err_pos, 1), "err_deg": round(float(np.degrees(result.err_yaw)), 2),
               "sim_s": round(result.seconds, 1), "base": [round(float(v), 3) for v in pose.pos]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        Image.fromarray(sim.capture(cam).rgb).save(out / f"{len(rows):02d}_{label}.png")

    first = True
    for label, (x, y) in points:
        bx, by = park_for(lay, x, y)
        wx, wy = a.to_world(bx, by)
        result = tuck_in(sim, wx, wy, np.radians(a.yaw_deg)) if first else shift_under(sim, wx, wy, np.radians(a.yaw_deg))
        first = False
        record(f"a_{label}", result, "a_overview")
    record("a_back_out", back_out(sim), "a_overview")
    bx, by = park_for(lay, *grid[len(grid) // 2])
    wx, wy = b.to_world(bx, by)
    record("b_middle", tuck_in(sim, wx, wy, np.radians(b.yaw_deg)), "b_overview")
    (out / "drive.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
