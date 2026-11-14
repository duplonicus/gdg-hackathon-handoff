"""Where do the MH3's fingertips land around a capsule as the hand closes? (Nov 5, to make the pick a real pinch.)

dup: "the leakers are floating away from the hand. attach them to the fingers and make the hand grasp as well." The
pick welds the capsule to the wrist with the hand open, hovering 4 mm over it, so nothing touches it. This parks at
cell a's station, puts a loose leaker on the left half of the tray, brings the left hand to the pick height over it,
then sets a series of finger poses and records, for each, where the index, middle and thumb tips are relative to the
capsule's centre (cell frame: x ahead, y left, z up), with a close-up picture. Nothing is welded.

    .\\run.ps1 scripts/probe_pinch.py --out output/cells/pinch
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_skills import HOVER, _to_local, park_station, reach_over  # noqa: E402
from handoff.sim.base import CameraSpec  # noqa: E402

TIPS = ("left_index_tip", "left_middle_tip", "left_thumb_tip")
POSES = {  # finger root joints (radians); the dependent joints follow by mimic
    "open": {},
    "index_0.6": {"left_index": 0.6},
    "index_1.0": {"left_index": 1.0},
    "index_1.4": {"left_index": 1.4},
    "all_1.0": {"left_index": 1.0, "left_middle": 1.0, "left_ring": 1.0, "left_little": 1.0},
    "thumb_1.2": {"left_thumb_1": 1.2},
    "thumb_1.2_0.45": {"left_thumb_1": 1.2, "left_thumb_2": 0.45},
    "thumb_1.6_0.45": {"left_thumb_1": 1.6, "left_thumb_2": 0.45},
    "pinch_1.0_1.2": {"left_index": 1.0, "left_thumb_1": 1.2, "left_thumb_2": 0.45},
    "pinch_1.4_1.6": {"left_index": 1.4, "left_thumb_1": 1.6, "left_thumb_2": 0.45},
}
FINGERS = ("left_index", "left_middle", "left_ring", "left_little", "left_thumb_1", "left_thumb_2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spot", type=int, default=None, help="grid spot for the leaker (left half of the tray); default: the left half's middle")
    parser.add_argument("--roll-deg", type=float, default=None, help="roll the hand about its pointing axis (90 = palm down for the left hand) and place the wrist by --wrist-offset instead of the pick move")
    parser.add_argument("--wrist-offset", default="-0.10,0.0,0.10", help="wrist from the capsule centre, cell frame (x ahead, y left, z up), m")
    parser.add_argument("--sweep", action="store_true", help="a grid of index x thumb closures (palm-down pinch), printing tip gap and midpoint")
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "pinch"))
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    cell, lay = cells.cell("a"), cells.layout
    if args.spot is None:
        left = [i for i, (_, y) in enumerate(lay.grid()) if y >= 0]
        args.spot = left[len(left) // 2]
    gx, gy = lay.grid()[args.spot]
    assert gy >= 0, "pick a spot on the left half"
    wx, wy = cell.to_world(gx, gy)
    z = lay.paper_top + lay.capsule[2] / 2.0
    side_eye = cell.to_world(gx + 0.10, gy - 0.16)
    front_eye = cell.to_world(gx + 0.22, gy + 0.02)
    scene = make_two_cell_scene(cells)
    scene = replace(scene, cameras=scene.cameras + (
        CameraSpec("pinch_side", pos=(*side_eye, z + 0.10), lookat=(wx, wy, z + 0.02), fov=40, res=(640, 400)),
        CameraSpec("pinch_front", pos=(*front_eye, z + 0.08), lookat=(wx, wy, z + 0.02), fov=40, res=(640, 400)),
    ))
    sim = GenesisSimulator(args.config)
    sim.build(scene)
    print(f"station: {park_station(sim, cell)}", flush=True)
    name = cells.leaker_names[0]
    sim.set_object_pose(cells.grid_names[args.spot], (wx, wy, -2.0), 0.0)
    sim.set_object_pose(name, (wx, wy, z), cell.yaw_deg)
    sim.settle(0.5)
    at = np.asarray(sim.object_state(name).pos, dtype=float)
    if args.roll_deg is None:
        err, heading, _ = reach_over(sim, "left", (at[0], at[1]), lay.paper_top + lay.capsule[2], HOVER, 0.005)
    else:
        from handoff.capsule_skills import hand_quat
        from handoff.sim.base import quat_mul, yaw_of

        r = np.radians(args.roll_deg)
        yaw = yaw_of(sim.base_pose().quat)
        roll = np.array([np.cos(r / 2), np.sin(r / 2), 0.0, 0.0])  # about the hand's pointing axis, before the base yaw
        from handoff.skills import HOME_QUAT
        from handoff.sim.base import quat_from_yaw

        quat = quat_mul(quat_from_yaw(yaw), quat_mul(roll, np.asarray(HOME_QUAT["left"], dtype=float)))
        dx, dy, dz = (float(v) for v in args.wrist_offset.split(","))
        target = np.array([*cell.to_world(gx + dx, gy + dy), float(at[2]) + dz])
        err, heading = float(sim.move_wrist("left", target, quat, timeout_s=4.0, tol_m=0.005)), f"roll {args.roll_deg}"
    print(f"hand placed: err {err * 1000:.1f} mm, {heading}", flush=True)

    def local(p) -> list[float]:
        u, v = _to_local(cell, float(p[0]), float(p[1]))
        cu, cv = _to_local(cell, float(at[0]), float(at[1]))
        return [round((u - cu) * 1000, 1), round((v - cv) * 1000, 1), round((float(p[2]) - float(at[2])) * 1000, 1)]

    poses = POSES
    if args.sweep:
        poses = {f"i{i:.1f}_t{t:.1f}": {"left_index": i, "left_thumb_1": t, "left_thumb_2": 0.45} for i in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4) for t in (1.2, 1.4, 1.6)}
    rows = []
    for label, pose in poses.items():
        sim.set_joints({f: pose.get(f, 0.0) for f in FINGERS})
        sim.settle(0.8)
        tips = {t: local(sim.kit.get_link_pose(t)[0]) for t in TIPS}
        wrist = local(sim.kit.get_link_pose("left_gripper_base")[0])
        moved = local(sim.object_state(name).pos)
        wpos = sim.kit.get_link_pose("left_gripper_base")[0]
        tips_from_wrist = {t: [round((float(a) - float(b)) * 1000, 1) for a, b in zip(sim.kit.get_link_pose(t)[0], wpos)] for t in TIPS}
        it, tt = (np.asarray(sim.kit.get_link_pose(t)[0], dtype=float) for t in ("left_index_tip", "left_thumb_tip"))
        if args.sweep:
            mid = (it + tt) / 2 - np.asarray(wpos, dtype=float)
            print(f"SWEEP {label}: tip gap {np.linalg.norm(it - tt) * 1000:.1f} mm, gap vector {np.round((it - tt) * 1000, 1).tolist()}, midpoint from wrist {np.round(mid * 1000, 1).tolist()} mm", flush=True)
        row = {"pose": label, "tips_from_wrist_world_mm": tips_from_wrist, "joints": pose, "tips_mm": tips, "wrist_mm": wrist, "capsule_moved_mm": moved}
        rows.append(row)
        print(json.dumps(row), flush=True)
        for cam in ("pinch_side", "pinch_front"):
            rgb = sim.capture(cam).rgb
            if rgb is not None:
                Image.fromarray(rgb).save(out / f"{label}_{cam}.png")
    (out / "pinch.json").write_text(json.dumps({"spot": args.spot, "heading": heading, "err_m": err, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
