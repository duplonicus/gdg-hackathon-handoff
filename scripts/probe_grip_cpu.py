"""Genesis CPU check of the grasp, one seed per process: drive, look, reach, approach, grasp, then lift and a 1 s hold
at the top (the carry's worst case for slip), with every fingertip read logged. The quick look behind the grip
findings in VALIDATION.md (Oct 21); scripts/probe_grip.py is the full before/after on the GPU.

Runs on gs.cpu with the decimated robot and the head camera off (no GPU), about a minute per seed:

    .\\run.ps1 python scripts/probe_grip_cpu.py --seed 0 --grip force
    .\\run.ps1 python scripts/probe_grip_cpu.py --seed 0 --grip fixed
    .\\run.ps1 python scripts/probe_grip_cpu.py --seed 0 --grip force --grip-force 6

Prints one JSON line per seed at the end (also written to output/probe/grip_cpu/<timestamp>/result.json): whether it
held, the closure where the tips first touched, the grip force, the keeper's tightening, the peak true penetration of
any hand link (truth, for the measurement only), and whether the part came up with the hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff import skills  # noqa: E402
from handoff.scene import Layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext, finger_side_n, grip_keeper  # noqa: E402


def cpu_config(out: Path) -> Path:
    """config/handoff.yaml on the CPU backend, with the decimated robot and every camera off."""
    cfg = yaml.safe_load((ROOT / "config" / "handoff.yaml").read_text())
    cfg["robot_usd"] = str((ROOT / "assets" / "MH3.usd").resolve())
    cfg["backend"] = "cpu"
    cfg["visual_face_ratio"] = 0.05
    for cam in cfg.get("cameras", {}).values():
        cam["enabled"] = False
    path = out / "cpu.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grip", choices=("force", "fixed"), default="force")
    parser.add_argument("--hold-s", type=float, default=1.0, help="hold at the top of the lift (s)")
    parser.add_argument("--grip-force", type=float, default=None, help="force grip target on the finger side (N); default the part budget")
    parser.add_argument("--substeps", type=int, default=1, help="physics substeps per step (stiffer contact)")
    parser.add_argument("--lead", type=float, default=None, help="force grip: how far the command may lead the fingers (closure)")
    parser.add_argument("--palm", type=float, default=None, help="force grip: palm skin force (N) required before the grip counts")
    parser.add_argument("--thumb", type=float, default=None, help="force grip: close the thumb on its own until its tip reads this (N)")
    parser.add_argument("--close", type=float, default=None, help="fixed grip: the closure to close to (default 0.9)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "grip_cpu" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from handoff.sim.genesis_sim import GenesisSimulator

    t0 = time.time()
    layout = Layout()
    sim = GenesisSimulator(str(cpu_config(out)), substeps=args.substeps)
    sim.build(make_scene(layout))
    episode = make_episode(layout, args.seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    part = episode.parts[0]
    ctx = TaskContext(layout=layout, target_part=part, parts=episode.parts, grasp_control=args.grip, grip_force_n=args.grip_force,
                      grip_palm_n=args.palm, grip_thumb_n=args.thumb)
    if args.lead is not None:
        ctx.grip_lead = args.lead
    if args.close is not None:
        ctx.close_fraction = args.close
    for name in ("drive_to_table", "look_at_table", "reach", "approach"):
        r = SKILLS[name](sim, ctx)
        if not r.ok:
            print(json.dumps({"seed": args.seed, "grip": args.grip, "failed_at": name, "reason": r.reason}))
            return 1

    reads: list[tuple[float, float, dict[str, float]]] = []
    real_read = sim.read_fingertips

    def logged(arm):
        tips = real_read(arm)
        reads.append((round(sim.time, 3), round(sim.hand_closure(arm), 4), {k.split("_")[1]: round(v, 3) for k, v in tips.items()}))
        return tips

    sim.read_fingertips = logged  # the grasp and the keeper read through here
    peak_pen = 0.0

    def penetration() -> float:
        return max([0.0, *sim.truth_finger_penetration("left", part).values()])

    grasp = SKILLS["grasp"](sim, ctx)
    peak_pen = max(peak_pen, penetration())
    result = {"seed": args.seed, "grip": args.grip, "grip_force_n": args.grip_force, "substeps": args.substeps, "lead": ctx.grip_lead, "palm_n": args.palm, "thumb_n": args.thumb, "close": ctx.close_fraction, "part": part, "grasp_ok": grasp.ok, "grasp_reason": grasp.reason,
              "grip_line": skills.grip_line(part, grasp),
              "first_touch_closure": next((c for _, c, t in reads if any(v > 0 for v in t.values())), None),
              "closure_after_grasp": round(sim.hand_closure("left"), 4), "pen_after_grasp_mm": round(1000 * penetration(), 2),
              "skin_after_grasp_n": {k.removeprefix("left_"): round(float(np.linalg.norm(v)), 2) for k, v in sim.read_tactile("left").items() if np.linalg.norm(v) > 0},
              "thumb": grasp.data.get("thumb"), "build_and_pick_wall_s": round(time.time() - t0, 1)}
    if grasp.ok:
        lift = SKILLS["lift"](sim, ctx)
        result.update(lift_ok=lift.ok, lift_reason=lift.reason)
        n = max(1, round(args.hold_s / sim.dt / 10))
        for _ in range(n):  # the hold at the top, sampling the truth penetration as it goes
            sim.step(10)
            peak_pen = max(peak_pen, penetration())
        tips = real_read("left")
        keeper = grip_keeper(sim, "left")
        result.update(
            held_after_hold=sim.hand_contacts("left", part) > 0 and float(sim.object_state(part).pos[2]) > layout.table_top + 0.05,
            finger_side_after_hold_n=round(finger_side_n(tips, "left"), 3), closure_after_hold=round(sim.hand_closure("left"), 4),
            keeper=keeper.summary() if keeper else None, pen_after_hold_mm=round(1000 * penetration(), 2),
        )
    result["peak_pen_mm"] = round(1000 * peak_pen, 2)
    result["wall_s"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    (out / "reads.json").write_text(json.dumps(reads))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
