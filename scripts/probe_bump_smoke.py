"""Genesis CPU smoke check of the bump detectors (src/handoff/bump.py), and the peaks of a bump-probe run.

**moves** (the default): a fixed pad under the left hand and a wall across its way, then three clean moves, a slow
press down onto the pad and a fast crossing into the wall. Every detector runs with its threshold out of reach and
the contact solver as the (never firing) decider, so nothing stops; each step's statistic is recorded with the
contact truth. Then a free-motion check of the torque model: the direct residual ``M dqdot/dt + bias - tau`` on the
arm's joints with the motor armature taken out of M (what ``GenesisSimulator.torque_model`` uses), with the full M,
and with the previous step's M. Backs config/bump.yaml's IMU threshold 3.0: on Oct 21 (one CPU run) the clean moves
peaked at 0.90 to 2.15 m/s^2 on the hand IMU (2.15 on a 6 cm lift from rest), the slow press at IMU 5.9 / torque 33 /
skin 6.3, the wall crossing at IMU 50 / torque 34 / skin 7.6; the residual's standard deviation with the armature
taken out was 0.12 to 0.17 N*m per joint (the full M: 1.43 on left_2), against the torque sensor's 0.1638.

**peaks** (``--peaks DIR``, no simulator): for each crossing that scripts/probe_bump_sensors.py recorded in DIR, each
detector's held statistic (config/bump.yaml's settings) before and after the first true contact (hand or carried
part). Behind bump.yaml's "clean crossings up to 1.25 and clipped ones 10 to 15" (a 2 seed x 2 speed CPU run of the
probe, Oct 21).

Runs on gs.cpu with the decimated robot and the cameras off (no GPU), about two minutes for the moves:

    .\\run.ps1 python scripts/probe_bump_smoke.py
    .\\run.ps1 python scripts/probe_bump_smoke.py --peaks output/probe/bump_sensors/<timestamp>

Writes the summary to output/probe/bump_smoke/<timestamp>/result.json (``--out`` for elsewhere) and prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402

from handoff import bump  # noqa: E402
from handoff.sim.base import BoxSpec, SceneSpec  # noqa: E402
from handoff.skills import HOME_QUAT  # noqa: E402

SENSORS = ("hand_imu", "torso_imu", "head_imu", "base_imu", "joint_torque", "tactile")
OFF = 1e9


def thresholds(cfg: bump.BumpConfig) -> dict[str, float]:
    return {n: (cfg.imu.threshold if "imu" in n else cfg.torque.threshold if n == "joint_torque" else cfg.tactile.threshold) for n in SENSORS}


def moves(out: Path) -> dict:
    from genesis.utils.misc import qd_to_torch

    from handoff.sim.genesis_sim import GenesisSimulator

    t0 = time.time()
    sim = GenesisSimulator(str(cpu_config(out)))
    sim.build(SceneSpec(boxes=(
        BoxSpec("pad", (0.30, 0.12, 0.10), (-0.75, 0.25, 0.75), fixed=True),  # top at 0.80, under the hand
        BoxSpec("wall", (0.40, 0.02, 0.20), (-0.78, 0.08, 0.88), fixed=True),  # across the hand's way toward -y
    )))
    print("built", round(time.time() - t0), "s", flush=True)
    base = bump.BumpConfig.from_yaml()
    thr = thresholds(base)
    sim.bump_config = replace(base, sensor="contact_truth", truth_every=1, imu=replace(base.imu, threshold=OFF),
                              torque=replace(base.torque, threshold=OFF), tactile=replace(base.tactile, threshold=OFF))
    rec: list[dict] = []
    real_step = bump.BumpGuard.step

    def recording_step(self):
        real_step(self)
        row = {n: d.stat for n, d in self.detectors.items()}
        tq = self.detectors.get("joint_torque")
        row["tq_res"] = None if tq is None or tq.residual is None else tq.residual.copy()
        rec.append(row)
        return False  # never stop: the contact truth decides, but must not end the move either

    def run(label, target, duration):
        rec.clear()
        sim.move_wrist("left", np.array(target), HOME_QUAT["left"], duration_s=duration, timeout_s=0.5, stop_on=("pad", "wall"))
        truth = [i for i, r in enumerate(rec) if r["contact_truth"] > 0]
        first = truth[0] if truth else None
        row = {"move": label, "steps": len(rec), "first_truth_step": first, "wrist_after": np.round(sim.wrist_pose("left").pos, 3).tolist(), "detectors": {}}
        for name in SENSORS:
            s = np.array([r.get(name, 0.0) for r in rec])
            pre = s[:first] if first is not None else s
            post = s[first : first + 30] if first is not None else np.zeros(0)
            lat = None
            if first is not None:
                over = np.nonzero(s[first:] > thr[name])[0]
                lat = None if len(over) == 0 else int(over[0])
            row["detectors"][name] = {"clean_peak": round(float(pre.max()) if len(pre) else 0.0, 3),
                                      "after_contact_peak": round(float(post.max()) if len(post) else 0.0, 3),
                                      "steps_to_config_threshold": lat}
        if first is not None and rec[first].get("tq_res") is not None:
            row["torque_residual_10_steps_after_contact"] = np.round(rec[min(first + 10, len(rec) - 1)]["tq_res"], 2).tolist()
        print(f"== {label}: {len(rec)} steps, first truth contact step {first}; "
              + ", ".join(f"{n} {d['clean_peak']}/{d['after_contact_peak']}" for n, d in row["detectors"].items()), flush=True)
        return row

    bump.BumpGuard.step = recording_step
    try:
        wrist = sim.wrist_pose("left").pos.copy()
        rows = [
            run("clean: up 6 cm", wrist + [0.0, 0.0, 0.06], 0.6),
            run("clean: forward 8 cm, fast", wrist + [0.08, 0.0, 0.06], 0.3),
            run("clean: back home", wrist, 0.8),
        ]
        sim.settle(0.5)
        rows.append(run("slow press down onto the pad", wrist + [0.0, 0.0, -0.10], 2.0))
        rows.append(run("lift", wrist + [0.0, 0.0, 0.0], 0.8))
        sim.settle(0.5)
        rows.append(run("fast crossing into the wall", wrist + [0.0, -0.25, 0.0], 0.6))
    finally:
        bump.BumpGuard.step = real_step

    # The torque model's free-motion consistency: residual with M minus the armature (used) against the full M.
    robot = sim.kit.robot
    dofs = sim._arm_dofs("left")
    armature = np.diag(sim._np(robot.get_dofs_armature()).reshape(-1))
    sim.move_wrist("left", wrist + [0.0, 0.0, 0.10], HOME_QUAT["left"], duration_s=0.5, timeout_s=0.3)
    prev_v = sim._np(robot.get_dofs_velocity()).reshape(-1)
    errs: dict[str, list] = {"M-armature": [], "M full": [], "M of previous step": []}
    prev_m = None
    for k in range(60):
        sim.set_joints({"left_2": float(0.2 * np.sin(k / 10.0))})
        sim.step(1)
        v = sim._np(robot.get_dofs_velocity()).reshape(-1)
        m = sim._np(robot.get_mass_mat())
        b = sim._np(qd_to_torch(sim.kit.scene.sim.rigid_solver.dyn_state.dofs.qf_bias, transpose=True)).reshape(-1)[robot.dof_start : robot.dof_start + robot.n_dofs]
        tau = sim.read_joint_torques("left")
        dv = v - prev_v
        errs["M-armature"].append(((m - armature)[dofs] @ dv) / sim.dt + b[dofs] - tau)
        errs["M full"].append((m[dofs] @ dv) / sim.dt + b[dofs] - tau)
        if prev_m is not None:
            errs["M of previous step"].append(((prev_m - armature)[dofs] @ dv) / sim.dt + b[dofs] - tau)
        prev_v, prev_m = v, m
    residual = {name: {"mean": np.round(np.mean(e, axis=0), 3).tolist(), "std": np.round(np.std(e, axis=0), 3).tolist()} for name, e in errs.items()}
    for name, r in residual.items():
        print(f"free motion direct residual, {name}: mean {r['mean']} std {r['std']}", flush=True)
    return {"mode": "moves", "config_thresholds": thr, "moves": rows, "free_motion_residual": residual,
            "torque_noise_sigma_spec": sim.sensors.torque.noise, "wall_s": round(time.time() - t0, 1)}


def peaks(probe_dir: Path) -> dict:
    import score_bump_sensors as score

    cfg = bump.BumpConfig.from_yaml()
    rows = []
    for t in score.load_trials(probe_dir):
        first = score.first_truth(t, "any")
        row = {"seed": t["seed"], "speed_mps": t.get("speed_mps"), "clip": bool(t["clip"]), "first_truth": first, "n_steps": t["n_steps"], "detectors": {}}
        for name in ("hand_imu", "head_imu", "torso_imu", "base_imu", "joint_torque", "tactile"):
            stats, hold = score.trace_stats(t, name, cfg, "any")
            h = bump.held(stats, hold)
            pre = h[:first] if first else h
            row["detectors"][name] = {"pre": round(float(pre.max()), 3) if len(pre) else 0.0, "pre_at": int(np.argmax(pre)) if len(pre) else None,
                                      "post": round(float(h[first:].max()), 3) if first is not None else None}
        rows.append(row)
        print(t["seed"], t.get("speed_mps"), "clip" if t["clip"] else "clean", "first", first, "n", t["n_steps"], " | ".join(
            f"{n} pre {d['pre']:.2f}@{d['pre_at']} post {d['post'] if d['post'] is None else round(d['post'], 2)}" for n, d in row["detectors"].items()), flush=True)
    clean = [r["detectors"]["hand_imu"]["pre"] for r in rows if not r["clip"]]
    clipped = [r["detectors"]["hand_imu"]["post"] for r in rows if r["clip"] and r["detectors"]["hand_imu"]["post"] is not None]
    return {"mode": "peaks", "probe_dir": str(probe_dir), "trials": rows,
            "hand_imu_clean_max": max(clean) if clean else None, "hand_imu_clipped_range": [min(clipped), max(clipped)] if clipped else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks", type=Path, default=None, help="a scripts/probe_bump_sensors.py output folder: report its peaks instead of running the moves")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "bump_smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    result = peaks(args.peaks) if args.peaks else moves(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("moves", "trials")}))
    print("wrote", out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
