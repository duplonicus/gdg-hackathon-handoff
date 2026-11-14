"""Where can each hand get to over a table, in the grasp orientation? (Nov 4, to place the capsule tray.)

The robot stands where the config spawns it, facing +x; targets are relative to its base (x ahead, y left), with no
table, so only the arm limits reach. Each arm visits a grid of wrist targets at one
height, in a snake order (each move is short), and the miss distance is recorded. Prints a map per arm
(``#`` reached within 1 cm, ``+`` within 2 cm, ``.`` missed) and writes every target to a JSON file.

    .\\run.ps1 scripts/probe_reach.py --z 0.88 --out output/bench/reach_z088.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.sim.base import SceneSpec  # noqa: E402



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--z", type=float, default=0.88, help="wrist height, m")
    parser.add_argument("--x", default="0.25,0.75,0.05", help="start,stop,step")
    parser.add_argument("--y", default="-0.60,0.60,0.10", help="start,stop,step")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="turn the grasp orientation about the vertical (+ = toward the left)")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="then tip it about the robot's left-right axis")
    parser.add_argument("--slider", type=float, default=None, help="torso lift, m (0 to 0.3); default: as the config spawns it")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "bench" / "reach.json"))
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator
    from handoff.sim.base import quat_from_yaw, quat_mul
    from handoff.skills import HOME_QUAT

    pitch = np.radians(args.pitch_deg)
    turn = quat_mul(quat_from_yaw(np.radians(args.yaw_deg)), np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]))

    def span(text: str) -> np.ndarray:
        a, b, s = (float(v) for v in text.split(","))
        return np.round(np.arange(a, b + s / 2, s), 3)

    xs, ys = span(args.x), span(args.y)
    sim = GenesisSimulator(args.config)
    sim.build(SceneSpec(boxes=(), robot_start=(0.0, 0.0, 0.0)))  # the kit spawns the robot from the config, not here
    if args.slider is not None:
        sim.set_joints({"slider": args.slider})
    sim.settle(0.5 if args.slider is None else 40.0)  # the lift moves at 8 mm/s
    base = np.asarray(sim.base_pose().pos, dtype=float)
    print(f"base at {np.round(base, 3).tolist()}", flush=True)
    rows = []
    for arm in ("left", "right"):
        quat = quat_mul(turn, np.asarray(HOME_QUAT[arm], dtype=float)) if arm == "left" else quat_mul(
            quat_mul(quat_from_yaw(-np.radians(args.yaw_deg)), np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])),
            np.asarray(HOME_QUAT[arm], dtype=float))  # the right hand mirrors the turn
        side = ys[ys >= -0.05] if arm == "left" else ys[ys <= 0.05]
        for i, x in enumerate(xs):
            order = side if i % 2 == 0 else side[::-1]
            for y in order:
                err = sim.move_wrist(arm, np.array([base[0] + x, base[1] + y, args.z]), quat, timeout_s=1.0, tol_m=0.005)
                row = {"arm": arm, "x": float(x), "y": float(y), "z": args.z, "err_m": round(float(err), 4)}
                rows.append(row)
                print(json.dumps(row), flush=True)  # progress, one line per target
        sim.reset()
        if args.slider is not None:
            sim.set_joints({"slider": args.slider})
        sim.settle(0.5 if args.slider is None else 40.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"z": args.z, "yaw_deg": args.yaw_deg, "pitch_deg": args.pitch_deg, "slider": args.slider, "base": base.tolist(), "frame": "x, y relative to the base, facing +x", "rows": rows}, indent=1), encoding="utf-8")
    for arm in ("left", "right"):
        print(f"\n{arm} arm, wrist z {args.z} m (rows x, columns y {ys[0]:.2f} to {ys[-1]:.2f}):")
        got = {(r["x"], r["y"]): r["err_m"] for r in rows if r["arm"] == arm}
        for x in xs:
            marks = "".join("#" if got.get((float(x), float(y)), 9) <= 0.01 else "+" if got.get((float(x), float(y)), 9) <= 0.02 else "." if (float(x), float(y)) in got else " " for y in ys)
            print(f"  x {x:.2f}  {marks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
