"""Build the two capsule cells and save what every scene camera sees, plus a wide view of the aisle (Nov 4).

For looking at the layout before any skill runs: is anything clipping, does a finished cell read as finished, can a
pool be seen from the tray camera. Optionally drops leakers into chosen grid spots first.

    .\\run.ps1 scripts/render_cells.py --out output/cells/look --leakers 40,133,201
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.sim.base import CameraSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "look"))
    parser.add_argument("--leakers", default="", help="grid indices in cell a to turn into leakers, comma separated")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    cells = TwoCells()
    spec = make_two_cell_scene(cells)
    wide = (
        CameraSpec("wide", pos=(-1.2, -3.2, 2.6), lookat=(-1.2, 0.0, 0.6), fov=60, res=(1280, 800)),
        CameraSpec("wide_back", pos=(-1.2, 3.2, 2.6), lookat=(-1.2, 0.0, 0.6), fov=60, res=(1280, 800)),
    )
    spec = replace(spec, cameras=spec.cameras + wide)  # keeps the scene's lamps, sun and floor (a fresh SceneSpec dropped them)
    sim = GenesisSimulator(args.config)
    began = time.perf_counter()
    sim.build(spec)
    print(f"built in {time.perf_counter() - began:.0f} s, {len(spec.boxes)} boxes", flush=True)
    lay, a = cells.layout, cells.cell("a")
    spots = [int(v) for v in args.leakers.split(",") if v.strip()]
    grid = lay.grid()
    for k, i in enumerate(spots):
        x, y = a.to_world(*grid[i])
        sim.set_object_pose(cells.grid_names[i], (x, y, -2.0))  # the fixed capsule there steps aside
        sim.set_object_pose(cells.leaker_names[k], (x, y, lay.paper_top + lay.capsule[2] / 2.0), a.yaw_deg)
        sim.set_object_pose(cells.pool_names[k], (x, y, lay.paper_top + 0.0003))
    sim.settle(1.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in spec.cameras:
        Image.fromarray(sim.capture(cam.name).rgb).save(out / f"{cam.name}.png")
    moved = {}
    for k in range(len(spots)):
        pos = sim.object_state(cells.leaker_names[k]).pos
        x, y = a.to_world(*grid[spots[k]])
        moved[cells.leaker_names[k]] = round(1000 * float(((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5), 2)
    # Perception check: find the capsules in cell a's tray camera and match them to where they truly are.
    import numpy as np

    from handoff.capsule_perception import crop_numbered, find_capsules

    rgb, xyz = sim.capture_cloud("a_tray")
    tray_xy = a.to_world(*lay.tray_center)
    seen = find_capsules(xyz, lay.paper_top, tray_xy, lay.paper_half, a.yaw_deg)
    truth = np.array([a.to_world(x, y) for x, y in grid])
    errs = [float(np.min(np.hypot(truth[:, 0] - cap.xyz[0], truth[:, 1] - cap.xyz[1]))) for cap in seen]
    Image.fromarray(crop_numbered(rgb, seen[:16])).save(out / "crop_16.png")
    perception = {"capsules_true": len(grid), "found": len(seen), "max_err_mm": round(1000 * max(errs), 2) if errs else None,
                  "median_err_mm": round(1000 * float(np.median(errs)), 2) if errs else None, "pixels_median": int(np.median([c.pixels for c in seen])) if seen else 0}
    print(json.dumps({"cameras": [c.name for c in spec.cameras], "leaker_drift_mm": moved, "perception": perception, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
