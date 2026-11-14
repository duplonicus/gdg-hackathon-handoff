"""Probe: what every sensor the robot carries reads when the hand, carrying a part, crosses a tray cleanly or clips it.

Genesis on the GPU (the config's backend); about 20 s of build, then roughly a minute per seed. Run it with the GPU to
spare (no d3 27B loaded):

    .\\run.ps1 python scripts/probe_bump_sensors.py --seeds 1-20
    .\\run.ps1 python scripts/score_bump_sensors.py output/probe/bump_sensors/<timestamp> --pick-seeds 1-10 --score-seeds 11-20

The scorer picks each sensor's threshold on the first seeds and scores it on the others (score_bump_sensors.py says
how); record what it prints in docs/VALIDATION.md with this command line and the commit.

**Per seed**: reset; the episode's parts and trays (``make_episode(seed)``, the trays nudged up to 2 cm as in the
demo); drive to the table; pick the first part (``--grasp-mode weld`` by default, the comparison arm, so every crossing
carries a part: a part dropped mid-probe would end that seed's crossings) and lift it. Then for each speed
(``--speeds``, m/s averaged over the path; the ease peaks at pi/2 of it), two crossings from above the emptied pick
slot to above the good tray's centre, straight over its near wall (``good_wall_ym``):

- **clean**: at the skills' crossing height (``skills.clearance_z``: wall tops + the held part's lower half + margin);
- **clip**: lowered on purpose, the wrist at the wall top plus an offset drawn per crossing from ``--clip-range``
  (default -0.02 to 0.035 m; the carried part's bottom hangs 4 cm under the wrist, so every clip reaches below the wall
  top, by 0.5 cm for the shallowest, a graze, up to 6 cm).

The probe aims with the trays' true positions (it must clip what it means to clip); the robot itself aims with what
its camera saw.

**Each crossing** uses the eased joint path straight to the goal's IK (GenesisSimulator.move_wrist's path until Oct 21;
move_wrist now follows the straight line, so traces recorded here are on the older, sagging path) and never stops on anything: it
records every physics step, runs on ``--after-contact-s`` along the path after the first true contact, then holds
where it is for ``--hold-s`` (a clean crossing settles at its goal that long instead), still recording. The hand then
goes up and back unrecorded. The contact truth is used for that cut-off and for scoring only, never by a robot decision.

**Written** to ``output/probe/bump_sensors/<timestamp>/`` (or ``--out``):

- ``trial_NNNN.npz`` per crossing, one row per physics step: ``t``, ``phase`` (0 moving, 1 holding), ``wrist`` (from
  the arm's kinematics), ``imu_acc`` and ``imu_gyro`` (step, placement, xyz), ``torque`` (7), ``dp`` and ``bias`` (7:
  the momentum observer's model terms, ``GenesisSimulator.torque_model``; NaN until ready), ``tactile`` (step, skin
  link, xyz), ``fingertips`` (step, 5), and the TRUTH columns ``truth_hand`` (contact points between the hand and the
  guarded boxes), ``truth_part`` (the carried part and the guarded boxes) and ``truth_box`` (index into ``boxes`` of the
  first box touched, -1 for none).
- ``trials.jsonl``, a line per crossing: seed, speed, clip, crossing and clip heights, the first true contact's step,
  the names behind every array axis (``placements``, ``skin_links``, ``fingertips``, ``boxes``), the commit, and the
  command line. A seed whose pick failed gets a ``"skipped"`` line with the reason.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from score_bump_sensors import parse_seeds  # noqa: E402

from handoff.scene import Layout, episode_layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.sim.base import fingertip_links, skin_links  # noqa: E402
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext, clearance_z, fixed_boxes, grasp_quat  # noqa: E402

PICK = ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift")
SAFE_ABOVE = 0.05  # m over the clearance height for the unrecorded moves between crossings


def commit() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


class Recorder:
    """Every sensor reading and the contact truth after one physics step, in fixed array orders."""

    def __init__(self, sim: Any, arm: str, boxes: tuple[str, ...], part: str) -> None:
        self.sim, self.arm, self.boxes, self.part = sim, arm, boxes, part
        self.placements = [p for p, _ in sim.sensors.placements]
        self.skin = list(skin_links(arm)) if arm in sim.sensors.arms("tactile") else []
        self.tips = list(fingertip_links(arm)) if arm in sim.sensors.arms("fingertip") else []
        if not self.placements or arm not in sim.sensors.arms("torque"):
            raise SystemExit("the probe needs the IMUs and the joint torque on (config/sensors.yaml)")
        self.owner = {int(link.idx): i for i, name in enumerate(boxes) for link in sim._entities[name].links}  # noqa: SLF001

    def _part_contacts(self) -> tuple[int, int]:
        """TRUTH: contact points between the carried part and the guarded boxes, and the first box's index (-1)."""
        contacts = self.sim._entities[self.part].get_contacts()  # noqa: SLF001
        a, b = contacts.get("link_a"), contacts.get("link_b")
        if a is None or b is None:
            return 0, -1
        n, first = 0, -1
        for la, lb in zip(self.sim._np(a).astype(int).ravel(), self.sim._np(b).astype(int).ravel(), strict=True):  # noqa: SLF001
            box = self.owner.get(int(lb), self.owner.get(int(la)))
            if box is not None:
                n += 1
                first = box if first < 0 else first
        return n, first

    def read(self, phase: int) -> dict[str, Any]:
        sim, arm = self.sim, self.arm
        imus = sim.read_imus()
        model = sim.torque_model(arm)
        skin = sim.read_tactile(arm)
        tips = sim.read_fingertips(arm)
        hand = sim.truth_hand_contacts(arm, self.boxes)
        part_n, part_box = self._part_contacts()
        hand_box = next((i for i, name in enumerate(self.boxes) if hand.get(name, 0) > 0), -1)
        return {
            "t": sim.time, "phase": phase, "wrist": sim.wrist_pose(arm).pos,
            "imu_acc": np.array([imus[p][0] for p in self.placements]), "imu_gyro": np.array([imus[p][1] for p in self.placements]),
            "torque": sim.read_joint_torques(arm),
            "dp": np.full(7, np.nan) if model is None else model[0], "bias": np.full(7, np.nan) if model is None else model[1],
            "tactile": np.array([skin[k] for k in self.skin]).reshape(len(self.skin), 3),
            "fingertips": np.array([tips[k] for k in self.tips], dtype=float),
            "truth_hand": int(sum(hand.values())), "truth_part": part_n, "truth_box": hand_box if hand_box >= 0 else part_box,
        }


def record_crossing(sim: Any, rec: Recorder, target: np.ndarray, quat: np.ndarray, duration_s: float, after_s: float, hold_s: float) -> list[dict[str, Any]]:
    """The eased joint path straight to the IK solution at ``target`` (GenesisSimulator.move_wrist before Oct 21: it now
    follows the straight line, so this path is the old one, which sags mid-crossing), recorded every step, never stopped by a
    contact: after the first true contact it runs on ``after_s`` along the path, then holds there ``hold_s`` (a clean
    crossing settles at its goal for ``hold_s``)."""
    from handoff.sim.genesis_sim import ARM_JOINTS, WRIST_LINK

    kit, arm = sim.kit, rec.arm
    joints = ARM_JOINTS[arm]
    solution = kit.solve_ik(WRIST_LINK[arm], position=np.asarray(target, dtype=float), orientation=quat, joints=joints, apply=False)
    q_idx = kit._public_q_indices  # noqa: SLF001  (read-only, as move_wrist does)
    goal = np.array([float(solution[q_idx[name]]) for name in joints])
    start = np.array([sim.joint_positions(joints)[name] for name in joints])
    n = max(1, int(round(duration_s / sim.dt)))
    after = int(round(after_s / sim.dt))
    rows: list[dict[str, Any]] = []
    first = None
    for k in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * k / n)
        kit.set_joint_positions(dict(zip(joints, start + s * (goal - start), strict=True)))
        sim.step(1)
        rows.append(rec.read(0))
        if first is None and (rows[-1]["truth_hand"] or rows[-1]["truth_part"]):
            first = len(rows) - 1
        if first is not None and len(rows) - 1 - first >= after:
            break
    if first is not None:
        # Hold where it is, as a guarded move does when it stops. A clean crossing keeps its goal instead, as move_wrist's
        # settle does: re-commanding the lagging measured position would jerk the arm back and read as a bump (an Oct 21
        # CPU smoke of this probe saw 5.2 m/s^2 on the hand IMU from exactly that).
        kit.set_joint_positions(sim.joint_positions(joints))
    for _ in range(int(round(hold_s / sim.dt))):
        sim.step(1)
        rows.append(rec.read(1))
    return rows


def save(out: Path, index: int, rows: list[dict[str, Any]]) -> tuple[str, int | None]:
    name = f"trial_{index:04d}.npz"
    arrays = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(out / name, **arrays)
    hits = np.nonzero((arrays["truth_hand"] > 0) | (arrays["truth_part"] > 0))[0]
    return name, (int(hits[0]) if len(hits) else None)


def go(sim: Any, arm: str, xyz, quat: np.ndarray) -> None:
    """An unrecorded, unguarded move (between crossings, high above everything)."""
    sim.move_wrist(arm, np.asarray(xyz, dtype=float), quat, timeout_s=1.5, tol_m=0.02)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-20", help="episode seeds, e.g. 1-20, or '1,4,9' (quoted: PowerShell splits a bare comma list)")
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.15, 0.25, 0.40], help="m/s along the crossing")
    parser.add_argument("--clip-range", type=float, nargs=2, default=[-0.02, 0.035], help="wrist height over the wall top, m, for the clip crossings")
    parser.add_argument("--after-contact-s", type=float, default=0.15)
    parser.add_argument("--hold-s", type=float, default=0.3)
    parser.add_argument("--grasp-mode", choices=("weld", "fingers"), default="weld")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"), help="the kit config (backend, dt)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    out = args.out or ROOT / "output" / "probe" / "bump_sensors" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    run = {**commit(), "command": " ".join(sys.argv), "config": args.config}

    from handoff.sim.genesis_sim import GenesisSimulator

    layout = Layout(bin_jitter_m=0.02)
    sim = GenesisSimulator(args.config)
    print("building the scene...")
    sim.build(make_scene(layout))
    index = 0
    with (out / "trials.jsonl").open("a", encoding="utf-8") as log:
        for seed in seeds:
            sim.reset()
            episode = make_episode(layout, seed)
            place_episode(sim, layout, episode)
            sim.set_joints(TRAVEL_POSE)
            sim.settle(1.0)
            truth_layout = episode_layout(layout, episode)
            part = episode.parts[0]
            ctx = TaskContext(layout=layout, target_part=part, parts=episode.parts, grasp_mode=args.grasp_mode,
                              bins_seen={name: (cx, cy) for name, cx, cy in truth_layout.bin_xy})
            failed = next(((name, r.reason) for name in PICK if not (r := SKILLS[name](sim, ctx)).ok), None)
            if failed is not None:
                log.write(json.dumps({"seed": seed, "skipped": True, "reason": f"{failed[0]}: {failed[1]}", **run}) + "\n")
                log.flush()
                print(f"seed {seed}: skipped ({failed[0]}: {failed[1]})")
                continue
            arm, quat, boxes = ctx.arm, grasp_quat(sim, ctx), fixed_boxes(ctx)
            rec = Recorder(sim, arm, boxes, part)
            region = truth_layout.bin_region("good")
            off = np.asarray(ctx.grasp_offset, dtype=float)
            (px, py, _), _yaw = episode.poses[part]
            begin = np.array([px + off[0], py + off[1]])
            end = np.array([region.cx + off[0], region.cy + off[1]])
            clean_z = clearance_z(sim, ctx)
            safe_z = clean_z + SAFE_ABOVE
            rng = np.random.default_rng([seed, 2027])
            for speed in args.speeds:
                for clip in (False, True):
                    offset = float(rng.uniform(*args.clip_range)) if clip else None
                    z = region.wall_top_z + offset if clip else clean_z
                    here = sim.wrist_pose(arm).pos
                    go(sim, arm, (here[0], here[1], safe_z), quat)
                    go(sim, arm, (begin[0], begin[1], safe_z), quat)
                    go(sim, arm, (begin[0], begin[1], z), quat)
                    sim.settle(0.3)
                    duration = float(np.linalg.norm(end - begin)) / speed
                    t0 = time.time()
                    rows = record_crossing(sim, rec, np.array([end[0], end[1], z]), quat, duration, args.after_contact_s, args.hold_s)
                    name, first = save(out, index, rows)
                    meta = {
                        "seed": seed, "trial": index, "file": name, "speed_mps": speed, "clip": clip, "crossing_z": round(z, 4),
                        "clip_offset_m": None if offset is None else round(offset, 4), "wall_top_z": round(region.wall_top_z, 4),
                        "first_truth_step": first, "n_steps": len(rows), "dt": sim.dt, "arm": arm, "part": part,
                        "grasp_mode": args.grasp_mode, "boxes": list(boxes), "placements": rec.placements, "skin_links": rec.skin,
                        "fingertips": rec.tips, **run,
                        # still carrying the part after the crossing (a fingers grasp can lose it on a clip)
                        "held_after": bool(np.linalg.norm(sim.object_state(part).pos - sim.wrist_pose(arm).pos) < 0.2),
                    }
                    log.write(json.dumps(meta) + "\n")
                    log.flush()
                    print(f"seed {seed} speed {speed:.2f} {'clip' if clip else 'clean'}: {len(rows)} steps, first contact {first}, {time.time() - t0:.1f} s")
                    index += 1
                    here = sim.wrist_pose(arm).pos
                    go(sim, arm, (here[0], here[1], safe_z), quat)
    print(f"wrote {index} crossings to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
