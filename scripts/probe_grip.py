"""Old fixed-closure grasp against the force-controlled one, on the same seeds: does stopping at the grip force keep
the fingers out of the part, and does the part still come up and ride to the bin?

    .\\run.ps1 python scripts\\probe_grip.py                        # 10 seeds (0..9), both grasps, config/handoff.yaml (GPU)
    .\\run.ps1 python scripts\\probe_grip.py --seeds 20 --seed0 100  # seeds 100..119
    .\\run.ps1 python scripts\\probe_grip.py --modes force           # one grasp only
    .\\run.ps1 python scripts\\probe_grip.py --seeds 20 --modes fixed,fixed@0.75   # the fixed closure at 0.9 vs 0.75
    .\\run.ps1 python scripts\\probe_grip.py --empty-ramp --config <a cpu config>   # the EMPTY_CLOSURE measurement only

--empty-ramp is the measurement behind skills.EMPTY_CLOSURE: the empty left hand (no part near it) ramped from 0.5 to
fully closed at the grip's slow rate (0.25 closure/s, a fingertip read every 2 steps); for each tip, the measured
closure (and command) at which it first reads anything. Oct 21, Genesis CPU backend (decimated robot, cameras off), one
run: index and thumb tips first at measured closure 0.818, command 0.845, 5.6 and 5.8 N; no other tip before 1.0.

GPU (cuda) through the Windows venv; never from WSL. About 30 s to build, then roughly 25 to 35 s of sim per trial on
the 3090 (each trial is the full pick-and-place: drive, reach, approach, grasp, lift, carry, lower, release, home).

Each trial is ``tasks.pick_place.run_pick_place`` on episode ``make_episode(layout, seed)``, the first part, with
``TaskContext.grasp_control`` "fixed" (close to 0.9 at once: the grasp up to Oct 21) or "force" (skills.grasp_force).
The rest of the context is the default for both, so the grasp is the only difference.

Per trial, scored from the simulator's truth (never seen by the robot):
- penetration: the deepest any hand link sinks into the part (``truth_finger_penetration``), sampled every 5 steps
  from the end of the approach to the end of the lift; the peak, and the value right after the grasp
- grasp_ok, the grasp's reason, its closure, the finger side's fingertip force, grasp time (sim s)
- held: the part came up with the hand (lift's ``held``); carry drop: held after the lift but not at the bin hover
- success: the part ended resting in the target bin (the trial's own checks)

Writes output/probe/grip/<timestamp>/: trials.jsonl (one line per trial), table.md (the comparison) and meta.json
(seeds, modes, config, git commit). Prints the table at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.scene import Layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import TRAVEL_POSE, TaskContext, grip_budget  # noqa: E402
from handoff.tasks.pick_place import run_pick_place  # noqa: E402

SAMPLE_EVERY = 5  # physics steps between penetration samples


def commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def step_data(record: dict, name: str) -> dict | None:
    return next((s for s in record["steps"] if s["skill"] == name), None)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    held = [r for r in rows if r["held"]]
    pen = [r["peak_penetration_mm"] for r in rows if r["peak_penetration_mm"] is not None]
    grasp_s = [r["grasp_s"] for r in rows if r["grasp_s"] is not None]
    return {
        "trials": n,
        "grasp_ok": sum(r["grasp_ok"] for r in rows),
        "held": len(held),
        "carry_drops": sum(r["carry_drop"] for r in rows),
        "success": sum(r["success"] for r in rows),
        "peak_pen_mean_mm": float(np.mean(pen)) if pen else None,
        "peak_pen_max_mm": float(np.max(pen)) if pen else None,
        "grasp_s_mean": float(np.mean(grasp_s)) if grasp_s else None,
        "closure_mean": float(np.mean([r["closure"] for r in rows if r["closure"] is not None])) if rows else None,
    }


def table(summary: dict[str, dict]) -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    lines = [
        "| grasp | trials | grasp ok | held (lift) | carry drops | in target bin | peak penetration mean / max (mm) | grasp time mean (s) | closure mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        lines.append(
            f"| {mode} | {s['trials']} | {s['grasp_ok']} | {s['held']} | {s['carry_drops']} | {s['success']} | "
            f"{f(s['peak_pen_mean_mm'], '.2f')} / {f(s['peak_pen_max_mm'], '.2f')} | {f(s['grasp_s_mean'], '.2f')} | {f(s['closure_mean'], '.2f')} |"
        )
    return "\n".join(lines)


def empty_ramp(sim, out: Path) -> dict:
    """Ramp the empty left hand from 0.5 to closed at the slow grip rate; where does each fingertip first read?"""
    from handoff.skills import GRIP_READ_EVERY, GRIP_SLOW_RATE, finger_side_n

    sim.set_hand("left", 0.5)
    sim.settle(0.5)
    cmd, first = 0.5, {}
    while cmd < 1.0:
        cmd = min(1.0, cmd + GRIP_SLOW_RATE * sim.dt * GRIP_READ_EVERY)
        sim.set_hand("left", cmd)
        sim.step(GRIP_READ_EVERY)
        tips = sim.read_fingertips("left")
        for tip, force in tips.items():
            if force > 0.0 and tip not in first:
                first[tip] = {"closure": sim.hand_closure("left"), "command": cmd, "force_n": force, "finger_side_n": finger_side_n(tips, "left")}
    sim.set_hand("left", 0.0)
    sim.settle(1.0)
    (out / "empty_ramp.json").write_text(json.dumps(first, indent=2), encoding="utf-8")
    for tip, row in first.items():
        print(f"first touch {tip}: measured closure {row['closure']:.3f}, command {row['command']:.3f}, {row['force_n']:.2f} N")
    print(f"empty ramp: {len(first)} of 5 tips touched something before full closure")
    return first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="first seed; trial k uses seed0 + k")
    parser.add_argument("--modes", default="fixed,force", help="comma list of grasp_control values; fixed@C closes the fixed grasp to C (default 0.9)")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=None, help="output folder (default output/probe/grip/<timestamp>)")
    parser.add_argument("--empty-ramp", action="store_true", help="only measure where the empty hand's tips first read (EMPTY_CLOSURE)")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    def valid(m: str) -> bool:
        control, _, close = m.partition("@")
        if control == "force":
            return not close
        try:
            return control == "fixed" and (not close or 0.0 < float(close) <= 1.0)
        except ValueError:
            return False

    if not modes or not all(valid(m) for m in modes):
        parser.error("--modes takes fixed, fixed@<closure> and/or force")

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "grip" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout()
    sim = GenesisSimulator(args.config)
    t0 = time.time()
    sim.build(make_scene(layout))
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    meta = {"seeds": seeds, "modes": modes, "config": args.config, "commit": commit(), "backend": str(sim.kit.compute_backend),
            "grip_budget": grip_budget(layout, sim.sensors.fingertip), "started": datetime.now().isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built in {time.time() - t0:.1f} s, backend {meta['backend']}, commit {meta['commit']}, results in {out}", flush=True)
    if args.empty_ramp:
        for name in layout.part_names:  # every part to the far park, nothing near the hand
            sim.set_object_pose(name, (3.0, 3.0 + 0.1 * layout.part_names.index(name), layout.part_size[2] / 2.0))
        sim.settle(0.5)
        empty_ramp(sim, out)
        return 0

    # Penetration sampling: on from the end of the approach ("pregrasp") to the end of the lift ("lifted").
    watch = {"on": False, "part": "", "steps": 0, "peak": 0.0, "peak_link": None}

    def sample(_simulation) -> None:
        if not watch["on"]:
            return
        watch["steps"] += 1
        if watch["steps"] % SAMPLE_EVERY:
            return
        depths = sim.truth_finger_penetration("left", watch["part"])
        if depths:
            link, depth = max(depths.items(), key=lambda kv: kv[1])
            if depth > watch["peak"]:
                watch["peak"], watch["peak_link"] = depth, link

    sim.kit.add_step_hook(sample)
    summary_rows: dict[str, list[dict]] = {m: [] for m in modes}
    with (out / "trials.jsonl").open("a", encoding="utf-8") as log:
        for seed in seeds:
            for mode in modes:
                sim.reset()
                episode = make_episode(layout, seed)
                place_episode(sim, layout, episode)
                sim.set_joints(TRAVEL_POSE)
                sim.settle(0.5)
                part = episode.parts[0]
                after_grasp: dict[str, float] = {}

                def on_event(tag: str, part: str = part, after_grasp: dict = after_grasp) -> None:
                    if tag == "pregrasp":
                        watch.update(on=True, part=part, steps=0, peak=0.0, peak_link=None)
                    elif tag == "closed":
                        after_grasp.update(sim.truth_finger_penetration("left", part))
                    elif tag in ("lifted", "failed_grasp", "failed_lift", "failed_approach"):
                        watch["on"] = False

                control, _, close = mode.partition("@")
                ctx = TaskContext(layout=layout, target_part=part, parts=episode.parts, grasp_control=control, on_event=on_event)
                if close:
                    ctx.close_fraction = float(close)
                watch.update(on=False, part=part, steps=0, peak=0.0, peak_link=None)  # nothing carried over from the last trial
                wall0 = time.time()
                result = run_pick_place(sim, ctx)
                watch["on"] = False
                record = result.as_dict()
                grasp_step, lift_step, carry_step = (step_data(record, n) for n in ("grasp", "lift", "move_to_bin"))
                g = grasp_step["data"] if grasp_step else {}
                held = bool(lift_step and lift_step["data"].get("held"))
                row = {
                    "mode": mode, "seed": seed, "part": part, "success": result.success, "failed_at": result.failed_at,
                    "grasp_ok": bool(grasp_step and grasp_step["ok"]), "grasp_reason": grasp_step["reason"] if grasp_step else None,
                    "grasp_s": (grasp_step["t_end"] - grasp_step["t_start"]) if grasp_step else None,
                    "closure": g.get("closure"), "closure_at_grip": g.get("closure_at_grip"), "closure_cmd": g.get("closure_cmd"),
                    "finger_side_n": g.get("finger_side_n"), "target_n": g.get("target_n"), "tips": g.get("tips"),
                    "held": held, "carry_drop": held and not bool(carry_step and carry_step["data"].get("held")),
                    "peak_penetration_mm": 1000.0 * watch["peak"] if grasp_step else None, "peak_link": watch["peak_link"],
                    "after_grasp_penetration_mm": {k: 1000.0 * v for k, v in after_grasp.items()},
                    "wall_s": round(time.time() - wall0, 1),
                }
                log.write(json.dumps({**row, "trial": record}) + "\n")
                log.flush()
                summary_rows[mode].append(row)
                pen = "n/a" if row["peak_penetration_mm"] is None else f"{row['peak_penetration_mm']:.2f} mm"
                print(f"[{mode}] seed {seed}: grasp {'ok' if row['grasp_ok'] else 'FAIL ' + str(row['grasp_reason'])}, "
                      f"held {held}, drop {row['carry_drop']}, success {result.success}, peak penetration {pen}, "
                      f"closure {row['closure']}, wall {row['wall_s']} s", flush=True)

    summary = {mode: summarise(rows) for mode, rows in summary_rows.items()}
    text = table(summary)
    (out / "table.md").write_text(
        f"# Grip probe {meta['started']}\n\nSeeds {seeds[0]}..{seeds[-1]}, commit {meta['commit']}, backend {meta['backend']}, "
        f"config {args.config}. Grip target {meta['grip_budget']['target_n']:.3f} N on the finger tips.\n\n{text}\n",
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
