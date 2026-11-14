"""What each MH3 joint turns, measured from the model: the basis of the dashboard's joint labels (Oct 21).

Builds the robot on the Genesis CPU backend (cameras off), then for each public joint, from the zero pose and from the
kit home pose: teleports that joint by +0.1 rad (no physics step), reads how the hand's root link (the head camera
mount for head joints, the joint's own tip for finger and thumb joints) rotated and moved, and reports the rotation
axis in the torso frame (torso_base: x to the robot's left, y backward, z up), or in the hand's frame (gripper_base)
for finger and thumb joints, plus each arm link's position.

    .\\run.ps1 scripts/probe_joint_axes.py            # output/probe/joint_axes/<ts>/axes.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402

from handoff.scene import Layout, make_scene  # noqa: E402

DELTA = 0.1  # rad


def rotation_axis(q_before: np.ndarray, q_after: np.ndarray) -> tuple[np.ndarray, float]:
    """World-frame axis and angle of the rotation that takes q_before to q_after (w, x, y, z)."""
    w0, v0 = q_before[0], -q_before[1:]
    w1, v1 = q_after[0], q_after[1:]
    w = w1 * w0 - v1 @ v0
    v = w1 * v0 + w0 * v1 + np.cross(v1, v0)
    if w < 0:
        w, v = -w, -v
    s = float(np.linalg.norm(v))
    return (v / s if s > 1e-9 else np.zeros(3)), 2.0 * float(np.arctan2(s, w))


def base_link(kit) -> str:
    return next(n for n in ("base_link", "base", "torso_base", "base_footprint") if n in kit.link_names)


def to_base(kit, vec: np.ndarray) -> np.ndarray:
    _, q = kit.get_link_pose(base_link(kit))
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r.T @ vec


def main() -> int:
    out = ROOT / "output" / "probe" / "joint_axes" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    from handoff.sim.genesis_sim import GenesisSimulator

    sim = GenesisSimulator(str(cpu_config(out)))
    sim.build(make_scene(Layout()))
    kit = sim.kit
    links = set(kit.link_names)
    names = [n for n in kit.joint_names if n != "slider"]
    home = dict(kit.home_positions)
    report: dict = {"links": sorted(links), "base_frame": base_link(kit), "joints": {}}
    print("base frame:", base_link(kit))
    chain = sorted(n for n in links if n.startswith(("left_arm_", "right_arm_"))) + [n for n in ("left_gripper_base", "right_gripper_base") if n in links]
    for pose_name, pose in (("zero", {n: 0.0 for n in names}), ("home", home)):
        # Where each arm link sits in this pose (base frame): a link's origin is where its parent joint turns.
        kit.set_joint_positions({n: float(np.clip(pose.get(n, 0.0), *[float(v[0]) for v in kit.get_joint_limits([n])])) for n in names}, immediate=True)
        report.setdefault("link_pos_base", {})[pose_name] = {}
        for link in chain:
            pos, _ = kit.get_link_pose(link, base_link(kit))
            report["link_pos_base"][pose_name][link] = [round(float(v), 3) for v in pos]
            print(f"{pose_name:4s} link {link:22s} at {np.round(pos, 3)}")
        for name in names:
            side = "left" if name.startswith("left_") else "right" if name.startswith("right_") else None
            probe_link = f"{side}_gripper_base" if side and f"{side}_gripper_base" in links else "head_camera_mount"
            finger = name.removeprefix(f"{side}_").split("_")[0] if side else ""
            in_hand = finger in ("index", "middle", "ring", "little", "thumb")
            if in_hand:  # a finger or thumb joint: watch its own tip, read in the hand's frame
                probe_link = f"{side}_{finger}_tip"
            lo, hi = (float(v[0]) for v in kit.get_joint_limits([name]))
            start = min(max(float(pose.get(name, 0.0)), lo), hi - DELTA)
            kit.set_joint_positions({n: float(np.clip(pose.get(n, 0.0), *[float(v[0]) for v in kit.get_joint_limits([n])])) for n in names}, immediate=True)
            kit.set_joint_positions({name: start}, immediate=True)
            frame = f"{side}_gripper_base" if in_hand else base_link(kit)
            p0, q0 = kit.get_link_pose(probe_link, frame)
            kit.set_joint_positions({name: start + DELTA}, immediate=True)
            p1, q1 = kit.get_link_pose(probe_link, frame)
            axis_b, angle = rotation_axis(np.asarray(q0, float), np.asarray(q1, float))
            entry = report["joints"].setdefault(name, {"limits_rad": [round(lo, 3), round(hi, 3)], "probe_link": probe_link, "frame": frame})
            moved = np.asarray(p1) - np.asarray(p0)
            entry[pose_name] = {"axis": [round(float(a), 3) for a in axis_b], "turned_rad": round(angle, 4),
                                "link_moved_mm": round(1000 * float(np.linalg.norm(moved)), 1), "link_moved_dir": [round(float(v), 2) for v in moved / max(float(np.linalg.norm(moved)), 1e-9)],
                                "link_pos": [round(float(v), 3) for v in p0]}
            print(f"{pose_name:4s} {name:22s} axis {np.round(axis_b, 2)} ({frame}) turned {angle:.3f} rad, {probe_link} at {np.round(p0, 3)} moved {entry[pose_name]['link_moved_mm']} mm along {entry[pose_name]['link_moved_dir']}")
    (out / "axes.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {out / 'axes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
