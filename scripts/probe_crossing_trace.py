"""Step-by-step trace of one carry's crossing on Genesis CPU: what jolts the hand IMU when nothing is touched?

Asked Oct 21: scripts/probe_carry.py (seed 7, closure 0.85) found the hand IMU stopping every crossing (move_to_bin and
retract) about 0.6 s in, at 18 to 26 m/s^2 against its 3.0 threshold, with the contact solver seeing no hand contact
with a tray or the table in 5 of 6 stops. This runs one part's pick and carry with the guard's stop switched off (the
detectors still run), and per physics step of move_to_bin and retract records the hand IMU's statistic and raw
acceleration, the wrist's position and speed, each arm joint's position, velocity and distance to its limits, the
commanded target, and every contact the robot's links have with anything (link names; contact solver, TRUTH used only
to explain the jolt).

    .\\run.ps1 python scripts/probe_crossing_trace.py --seed 7 --part 2 --close 0.85

Writes output/probe/crossing_trace/<timestamp>/trace.json and prints the steps around each IMU peak.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402

from handoff.scene import Layout, episode_layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext  # noqa: E402

TRACED = ("move_to_bin", "retract")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--part", type=int, default=2, help="index of the part in the episode")
    parser.add_argument("--close", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "crossing_trace" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from handoff.sim.genesis_sim import ARM_JOINTS, GenesisSimulator

    sim = GenesisSimulator(str(cpu_config(out)))
    layout = Layout()
    sim.build(make_scene(layout))
    kit = sim.kit
    names = {}
    for ent in kit.scene.entities:
        for link in ent.links:
            names[int(link.idx)] = link.name
    for obj, ent in sim._entities.items():  # noqa: SLF001
        for link in ent.links:
            names[int(link.idx)] = obj
    arm = "left"
    joints = ARM_JOINTS[arm]
    limits = {j: tuple(float(v[0]) for v in kit.get_joint_limits([j])) for j in joints}

    episode = make_episode(layout, args.seed)
    part = episode.parts[args.part]
    target = "rework" if episode.truth(part).defective else "good"
    place_episode(sim, layout, episode)
    sim.set_object_pose(part, layout.slot_pos(0), 0.0)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(1.0)
    truth = episode_layout(layout, episode)
    ctx = TaskContext(layout=layout, target_part=part, parts=episode.parts, target_bin=target, grasp_mode="fingers", close_fraction=args.close,
                      bins_seen={n: (cx, cy) for n, cx, cy in truth.bin_xy})
    sim.bins_seen = ctx.bins_seen

    rows: list[dict] = []
    state = {"skill": None, "guard": None, "prev_wrist": None}
    real_stopped, real_step = sim._stopped_on, sim.step  # noqa: SLF001

    def stopped_on(arm_, guard):  # the detectors run, the arm never stops: the whole crossing is traced
        state["guard"] = guard
        real_stopped(arm_, guard)
        return False

    def step(n=1):
        for _ in range(n):
            real_step(1)
            if state["skill"] not in TRACED:
                continue
            wrist = sim.wrist_pose(arm).pos.copy()
            speed = 0.0 if state["prev_wrist"] is None else float(np.linalg.norm(wrist - state["prev_wrist"]) / sim.dt)
            state["prev_wrist"] = wrist
            q = sim.joint_positions(joints)
            qd = sim._np(kit.robot.get_dofs_velocity()).reshape(-1)[sim._arm_dofs(arm)]  # noqa: SLF001
            contacts = kit.robot.get_contacts()
            la, lb = contacts.get("link_a"), contacts.get("link_b")
            pairs = []
            if la is not None and lb is not None:
                for a, b in zip(sim._np(la).astype(int).ravel(), sim._np(lb).astype(int).ravel(), strict=True):  # noqa: SLF001
                    pairs.append(f"{names.get(int(a), a)}~{names.get(int(b), b)}")
            guard = state["guard"]
            det = guard.detectors.get("hand_imu") if guard is not None and guard.active else None
            acc = sim.read_imus()["hand"][0] if "hand" in sim.read_imus() else None
            rows.append({
                "skill": state["skill"], "t": round(sim.time, 3), "wrist": [round(float(v), 4) for v in wrist], "wrist_speed": round(speed, 3),
                "imu_stat": None if det is None else round(det.stat, 2), "acc": None if acc is None else [round(float(v), 2) for v in acc],
                "q": {j: round(float(q[j]), 4) for j in joints},
                "qd": [round(float(v), 3) for v in qd],
                "near_limit": {j: round(min(q[j] - limits[j][0], limits[j][1] - q[j]), 4) for j in joints if min(q[j] - limits[j][0], limits[j][1] - q[j]) < 0.05},
                "contacts": sorted(set(pairs)),
            })

    sim._stopped_on, sim.step = stopped_on, step  # noqa: SLF001
    for name in ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift", "move_to_bin", "lower", "release", "retract"):
        state["skill"], state["prev_wrist"] = name, None
        r = SKILLS[name](sim, ctx)
        print(f"{name}: ok={r.ok} {r.reason}", flush=True)
    (out / "trace.json").write_text(json.dumps({"seed": args.seed, "part": part, "close": args.close, "limits": limits, "rows": rows}, indent=0), encoding="utf-8")

    for skill in TRACED:
        sk = [r for r in rows if r["skill"] == skill and r["imu_stat"] is not None]
        if not sk:
            continue
        peak = max(range(len(sk)), key=lambda i: sk[i]["imu_stat"])
        print(f"\n{skill}: {len(sk)} guarded steps, IMU peak {sk[peak]['imu_stat']} at t {sk[peak]['t']}")
        for r in sk[max(0, peak - 8): peak + 5]:
            print(f"  t {r['t']:.2f} imu {r['imu_stat']:6.2f} acc {r['acc']} wrist {r['wrist']} v {r['wrist_speed']:.2f} near_limit {r['near_limit']} qd {r['qd']} contacts {r['contacts']}")
    print(f"written {out / 'trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
