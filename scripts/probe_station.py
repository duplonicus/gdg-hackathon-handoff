"""Can the MH3 work the whole capsule tray from ONE park spot, turning its hands instead of driving? (Nov 5)

dup: cut all the driving inside a cell. The Nov 4 reach probe tried one hand heading (fingers forward) and found a
band too shallow for the 35 cm tray, so the robot drove for every leaker. The wrist's own reach is about 0.15 to
0.45 m ahead and 0.1 to 0.5 m to its side (probe_reach.py, also with the hand turned 45 degrees), but the hand's
working point sits 12 cm past the wrist, so turning the hand moves that point: fingers forward reach the far rows,
fingers pointing inward reach the middle columns and the near rows.

This tucks in once at cell a's station (base origin on the cell origin), then for every capsule spot tries the arm
on its side with hand headings from straight ahead to pointing inward, first the lift height then the pick height,
and keeps the first heading that hits within 5 mm. Then the same for candidate leaker-bin spots beside the tray.
Nothing is picked; this is reach in the real cell (table, tray, legs), not a pick test.

    .\\run.ps1 scripts/probe_station.py --out output/cells/station
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

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_skills import HAND_BELOW_WRIST, HAND_POINT, HOVER, LIFT, _to_local  # noqa: E402
from handoff.drive import tuck_in  # noqa: E402
from handoff.sim.base import quat_from_yaw, quat_mul, yaw_of  # noqa: E402
from handoff.skills import HOME_QUAT, TRAVEL_POSE  # noqa: E402

HEADINGS_DEG = (0, -30, -60, -90)  # for the left hand; negative turns the fingers toward the robot's middle


def targets(point_xy, top_z: float, base_yaw: float, arm: str, heading_deg: float):
    """Wrist positions (lift, pick) and the hand quaternion that put ``arm``'s working point over ``point_xy``."""
    side = 1.0 if arm == "left" else -1.0
    h = np.radians(heading_deg) * side
    yaw = base_yaw + h
    hp = np.array([HAND_POINT[0], HAND_POINT[1] * side])
    c, s = np.cos(yaw), np.sin(yaw)
    off = np.array([c * hp[0] - s * hp[1], s * hp[0] + c * hp[1]])
    xy = np.asarray(point_xy, dtype=float) - off
    quat = quat_mul(quat_from_yaw(h), quat_mul(quat_from_yaw(base_yaw), np.asarray(HOME_QUAT[arm], dtype=float)))
    lift = np.array([xy[0], xy[1], top_z + LIFT + HAND_BELOW_WRIST])
    pick = np.array([xy[0], xy[1], top_z + HOVER + HAND_BELOW_WRIST])
    return lift, pick, quat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "station"))
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--every", type=int, default=1, help="try every n-th grid spot (1 = all)")
    parser.add_argument("--fresh", action="store_true", help="each spot from the travel pose with the shift's own reach_over, as a shift does (the default walks spot to spot, which overstates reach: Nov 5)")
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    cell = cells.cell("a")
    lay = cell.layout
    sim = GenesisSimulator(args.config)
    sim.build(make_two_cell_scene(cells))
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    t0 = time.time()
    wx, wy = cell.to_world(0.0, 0.0)
    park = tuck_in(sim, wx, wy, float(np.radians(cell.yaw_deg)))
    if args.fresh:
        sim.brake(True, f"{cell.name}_table")  # as park_station does: the base cannot be shoved
    base_yaw = yaw_of(sim.base_pose().quat)
    print(f"tucked in: {park}", flush=True)
    top_z = lay.paper_top + lay.capsule[2]
    rows = []

    def attempt(kind: str, local_xy, z: float, arm: str) -> dict:
        world = cell.to_world(*local_xy)
        tried = []
        for heading in HEADINGS_DEG:
            lift, pick, quat = targets(world, z, base_yaw, arm, heading)
            e_lift = float(sim.move_wrist(arm, lift, quat, timeout_s=2.0, tol_m=0.01))
            e_pick = float(sim.move_wrist(arm, pick, quat, timeout_s=2.0, tol_m=0.005)) if e_lift <= 0.02 else None
            sim.move_wrist(arm, lift, quat, timeout_s=1.0, tol_m=0.02)
            tried.append({"heading_deg": heading, "err_lift": round(e_lift, 4), "err_pick": None if e_pick is None else round(e_pick, 4)})
            if e_pick is not None and e_pick <= 0.005:
                break
        ok = tried[-1]["err_pick"] is not None and tried[-1]["err_pick"] <= 0.005
        row = {"kind": kind, "local": [round(float(v), 4) for v in local_xy], "arm": arm, "ok": ok, "heading_deg": tried[-1]["heading_deg"] if ok else None, "tried": tried}
        print(json.dumps(row), flush=True)
        return row

    grid = lay.grid()
    for i, (x, y) in enumerate(grid):
        if i % args.every:
            continue
        if args.fresh:
            from handoff.capsule_skills import reach_over, side_of, travel_pose

            travel_pose(sim)
            arm = side_of(y)
            from handoff.capsule_skills import PICK_HEADINGS

            centre = lay.paper_top + lay.capsule[2] / 2.0  # the pinch goes to the capsule's centre (palm-down, Nov 5)
            err, heading, _ = reach_over(sim, arm, cell.to_world(x, y), centre, HOVER, 0.005, PICK_HEADINGS)
            bp = sim.base_pose()
            bu, bv = _to_local(cell, float(bp.pos[0]), float(bp.pos[1]))
            row = {"base_mm": [round(bu * 1000, 1), round(bv * 1000, 1), round(float(np.degrees(yaw_of(bp.quat) - base_yaw)), 2)],
                   "kind": "capsule", "spot": i, "local": [round(float(x), 4), round(float(y), 4)], "arm": arm, "ok": bool(err <= 0.005),
                   "heading_deg": -heading if err <= 0.005 else None, "err": round(float(err), 4), "tried": []}
            print(json.dumps(row), flush=True)
            rows.append(row)
            continue
        rows.append(attempt("capsule", (x, y), top_z, "left" if y >= 0 else "right") | {"spot": i})
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.6)
    rim = lay.table_top + lay.leaker_bin_floor + lay.leaker_bin_wall
    for bx in (0.25, 0.32, 0.40):
        for by in (0.30, 0.36):
            for arm, sy in (("left", 1.0), ("right", -1.0)):
                rows.append(attempt("bin", (bx, by * sy), rim, arm))
    caps = [r for r in rows if r["kind"] == "capsule"]
    summary = {
        "park": str(park), "base_yaw": base_yaw, "wall_s": round(time.time() - t0, 1),
        "capsules_tried": len(caps), "capsules_ok": sum(r["ok"] for r in caps),
        "by_heading": {str(h): sum(1 for r in caps if r["ok"] and r["heading_deg"] == h) for h in HEADINGS_DEG},
        "missed": [r["spot"] for r in caps if not r["ok"]],
        "bins_ok": [r["local"] + [r["arm"], r["heading_deg"]] for r in rows if r["kind"] == "bin" and r["ok"]],
    }
    (out / "station.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    # a map of the tray, far row on top: L/R = that arm reached it (letter = heading index), . = missed
    xs = sorted({round(x, 4) for x, _ in grid}, reverse=True)
    ys = sorted({round(y, 4) for _, y in grid}, reverse=True)
    got = {(round(r["local"][0], 4), round(r["local"][1], 4)): r for r in caps}
    for x in xs:
        line = ""
        for y in ys:
            r = got.get((x, y))
            line += " " if r is None else ("." if not r["ok"] else str(HEADINGS_DEG.index(r["heading_deg"])) if r["heading_deg"] in HEADINGS_DEG else "?")
        print(f"  x {x:.3f}  {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
