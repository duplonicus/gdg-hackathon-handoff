"""Pick leakers off a tray and drop them in the leaker bin, in Genesis (Nov 4).

Puts loose leakers (with their pools) at chosen grid spots in cell a, then for each: park for it, weld-pick it,
park for the leaker bin, drop it. Records whether each step worked, where the capsule ended up (truth, for this
check only), how long it took, and saves the tray and overview cameras after each pick and drop.

    .\\run.ps1 scripts/probe_pick_leaker.py --spots 30,3,60 --out output/cells/pick
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
from handoff.capsule_skills import drop_in_bin, park_at, pick_capsule, _to_local  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spots", default="30,3,60")  # Nov 4 one-hand flow (park_at); spots on the 64-capsule grid
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "pick"))
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    cells = TwoCells()
    sim = GenesisSimulator(args.config)
    sim.build(make_two_cell_scene(cells))
    a, lay = cells.cell("a"), cells.layout
    grid = lay.grid()
    spots = [int(v) for v in args.spots.split(",")]
    for k, i in enumerate(spots):
        x, y = a.to_world(*grid[i])
        sim.set_object_pose(cells.grid_names[i], (x, y, -2.0))
        sim.set_object_pose(cells.leaker_names[k], (x, y, lay.paper_top + lay.capsule[2] / 2.0), a.yaw_deg)
        sim.set_object_pose(cells.pool_names[k], (x, y, lay.paper_top + 0.0003))
    sim.settle(0.5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bin_xy = a.to_world(*lay.leaker_bin("left"))
    floor = lay.table_top + lay.leaker_bin_floor
    rows = []
    for k, i in enumerate(spots):
        name = cells.leaker_names[k]
        t0 = sim.time
        row = {"spot": i, "leaker": name}
        pos = sim.object_state(name).pos
        steps = [
            ("park_capsule", lambda: park_at(sim, a, _to_local(a, float(pos[0]), float(pos[1])), first=(k == 0))),
            ("pick", lambda: pick_capsule(sim, a, name, lay.paper_top + lay.capsule[2])),
            ("park_bin", lambda: park_at(sim, a, lay.leaker_bin("left"))),
            ("drop", lambda: drop_in_bin(sim, a, name)),
        ]
        for label, step in steps:
            result = step()
            row[label] = {"ok": result.ok, "reason": result.reason, "err_mm": round(1000 * result.err, 1)}
            if label in ("pick", "drop"):
                for cam in ("a_tray", "a_overview", "a_leaker_bin_left"):
                    Image.fromarray(sim.capture(cam).rgb).save(out / f"{k}_{label}_{cam}.png")
            if not result.ok:
                break
        end = np.array(sim.object_state(name).pos)
        row["end"] = [round(float(v), 4) for v in end]
        row["in_bin"] = bool(abs(end[0] - bin_xy[0]) <= lay.leaker_bin_inner[0] / 2 and abs(end[1] - bin_xy[1]) <= lay.leaker_bin_inner[1] / 2 and floor <= end[2] <= floor + lay.leaker_bin_wall)
        row["sim_s"] = round(sim.time - t0, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "pick.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
