"""Why does a hand crawl toward the tray's outer column from the forearms-up pose? (Nov 5 diagnostic)

Station shifts missed picks on the left hand's outer column (spots 254, 258, 259) and repeated moves closed in only a
few centimetres each (51, 42, 31 mm). This makes one such move and prints, flat: the error; whether move_wrist fell
back from its straight line (line_fallback_log); the commanded joint goal vs the joints reached (a gap = something
stops the arm: a contact or a limit; no gap = the IK goal itself is off); the joint limits hit; and every contact
between the robot and the scene at the end of the move.

    .\\run.ps1 scripts/probe_reach_stall.py --cell a --spot 258 --arm left --heading 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_skills import LIFT, hand_quat, park_station, travel_pose, wrist_for  # noqa: E402
from handoff.sim.base import yaw_of  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell", default="a")
    parser.add_argument("--spot", type=int, default=None, help="grid spot; default: the far corner on the arm's side")
    parser.add_argument("--arm", default="left")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--lift", type=float, default=LIFT, help="height of the move's target above the capsule tops, m")
    parser.add_argument("--via", default=None, help="x,y (cell frame): go to LIFT over this point first, heading 0")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    args = parser.parse_args()

    from handoff.sim.genesis_sim import ARM_JOINTS, WRIST_LINK, GenesisSimulator

    cells = TwoCells()
    cell, lay = cells.cell(args.cell), cells.layout
    if args.spot is None:
        side = [i for i, (_, y) in enumerate(lay.grid()) if (y >= 0) == (args.arm == "left")]
        args.spot = max(side, key=lambda i: (lay.grid()[i][0], abs(lay.grid()[i][1])))
    sim = GenesisSimulator(args.config)
    sim.build(make_two_cell_scene(cells))
    print("park", park_station(sim, cell), flush=True)
    travel_pose(sim)
    yaw = yaw_of(sim.base_pose().quat)
    pt = cell.to_world(*lay.grid()[args.spot])
    top = lay.paper_top + lay.capsule[2]
    quat = hand_quat(yaw, args.arm, args.heading)
    target = wrist_for(pt, top, yaw, args.lift, args.arm, args.heading)
    joints = ARM_JOINTS[args.arm]
    kit = sim.kit
    q_idx = kit._public_q_indices  # noqa: SLF001
    goal = kit.solve_ik(WRIST_LINK[args.arm], position=target, orientation=quat, joints=joints, apply=False)
    goal = np.array([float(goal[q_idx[n]]) for n in joints])
    before = len(sim.line_fallback_log or [])
    if args.via:
        vx, vy = (float(v) for v in args.via.split(","))
        via = cell.to_world(vx, vy)
        e = sim.move_wrist(args.arm, wrist_for(via, top, yaw, args.lift, args.arm, 0.0), hand_quat(yaw, args.arm, 0.0), timeout_s=3.0, tol_m=0.01)
        print(f"via {args.via}: err {e * 1000:.1f} mm", flush=True)
    def base_line(tag):
        from handoff.capsule_skills import _to_local

        bp = sim.base_pose()
        u, v = _to_local(cell, float(bp.pos[0]), float(bp.pos[1]))
        print(f"base {tag}: x {u * 1000:+.1f} mm, y {v * 1000:+.1f} mm, yaw {np.degrees(yaw_of(bp.quat)) - cell.yaw_deg:+.2f} deg", flush=True)

    base_line("before the reach")
    for k in range(3):
        err = sim.move_wrist(args.arm, target, quat, timeout_s=3.0, tol_m=0.01)
        base_line(f"after try {k + 1}")
        reached = np.array([sim.joint_positions(joints)[n] for n in joints])
        print(f"try {k + 1}: err {err * 1000:.1f} mm; wrist at {np.round(sim.wrist_pose(args.arm).pos, 3).tolist()} target {np.round(target, 3).tolist()}", flush=True)
        print("  fresh IK goal  " + " ".join(f"{v:+.2f}" for v in goal), flush=True)
        print("  joints reached " + " ".join(f"{v:+.2f}" for v in reached), flush=True)
    from handoff.capsule_skills import HOVER

    down = wrist_for(pt, top, yaw, HOVER, args.arm, args.heading)
    for k in range(2):
        err = sim.move_wrist(args.arm, down, quat, timeout_s=3.0, tol_m=0.005)
        base_line(f"after descent {k + 1}")
        print(f"descent {k + 1}: err {err * 1000:.1f} mm; wrist at {np.round(sim.wrist_pose(args.arm).pos, 3).tolist()} target {np.round(down, 3).tolist()}", flush=True)
        c = kit.robot.get_contacts()
        la, lb = c.get("link_a"), c.get("link_b")
        names = {int(link.idx): n for n, ent in sim._entities.items() for link in ent.links}  # noqa: SLF001
        rl = {int(link.idx): link.name for link in kit.robot.links}
        pairs = set()
        if la is not None:
            for a, b in zip(np.asarray(sim._np(la)).astype(int).ravel(), np.asarray(sim._np(lb)).astype(int).ravel(), strict=True):  # noqa: SLF001
                pairs.add((rl.get(a) or names.get(a, a), rl.get(b) or names.get(b, b)))
        print("  contacts:", sorted(str(p) for p in pairs if not all(str(x).startswith(("a_", "b_", "capsule", "leaker", "spare", "pool")) for x in p))[:12], flush=True)
    lims = sim.kit.get_joint_limits(list(joints)) if hasattr(sim.kit, "get_joint_limits") else None
    if lims is not None:
        lo, hi = (np.asarray(v, dtype=float).ravel() for v in (lims if isinstance(lims, tuple) else (lims[0], lims[1])))
        print("  limits low     " + " ".join(f"{v:+.2f}" for v in lo), flush=True)
        print("  limits high    " + " ".join(f"{v:+.2f}" for v in hi), flush=True)
    print("line fallbacks this move:", (sim.line_fallback_log or [])[before:], flush=True)
    contacts = kit.robot.get_contacts()
    la, lb = contacts.get("link_a"), contacts.get("link_b")
    names = {}
    for ent_name, ent in sim._entities.items():  # noqa: SLF001
        for link in ent.links:
            names[int(link.idx)] = ent_name
    robot_links = {int(link.idx): link.name for link in kit.robot.links}
    seen = set()
    if la is not None:
        for a, b in zip(np.asarray(sim._np(la)).astype(int).ravel(), np.asarray(sim._np(lb)).astype(int).ravel(), strict=True):  # noqa: SLF001
            pair = (robot_links.get(a) or names.get(a, a), robot_links.get(b) or names.get(b, b))
            seen.add(pair)
    print("contacts:", sorted(str(p) for p in seen if any(isinstance(x, str) and (x.startswith("left") or x.startswith("right")) for x in p)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
