"""Genesis CPU check of a whole pick and carry: does the fixed grasp hold through the crossing, and where does a cycle's time go?

Asked Oct 21 after the demo (seed 7, ``--close 0.75``) dropped part_12 on the way to the rework tray on every try
(a grip with the thumb at 0 N), and the arm moved in many short steps with pauses between them. probe_grip_cpu.py
tests only the lift and a 1 s hold; this runs the demo's carry: reach, approach, grasp, lift, move_to_bin, lower,
release, retract (after drive_to_table and look_at_table, untimed), for each part of an episode, on the target tray the part's truth says (defective: rework).

Per run it records each skill's result, the skill in which the part left the hand (contact solver: truth, used only to
measure), each fingertip's force after the grasp, the fingers' deepest penetration into the part after the grasp, lift, move_to_bin and lower, the hand's heading after each skill and its largest tilt from level during each skill, every bump the guard stopped on (its report, and the skill's ctx.bumps), and for every wrist move its planned (eased) time and the settle
tail after it (sim seconds), plus each settle() call. Runs on gs.cpu with the decimated robot and cameras off:

    .\\run.ps1 python scripts/probe_carry.py --seed 7 --closes 0.75,0.85,0.9
    .\\run.ps1 python scripts/probe_carry.py --seed 7 --grip force --grasp-mode weld   # touch, then weld

The summary goes to output/probe/carry/<timestamp>/result.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402

from handoff.checks import snapshot  # noqa: E402
from handoff.scene import Layout, episode_layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext, part_in_hand  # noqa: E402

SETUP = ("drive_to_table", "look_at_table")  # untimed: the robot drives up and looks at the table first
FLOW = ("reach", "approach", "grasp", "lift", "move_to_bin", "lower", "release", "retract")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds", default=None, help="several episodes, e.g. 1-10 (overrides --seed; quote a comma list in PowerShell)"
    )
    parser.add_argument(
        "--keep-keeper",
        action="store_true",
        help="weld: leave the grip keeper running after the weld (the default stops it)",
    )
    parser.add_argument(
        "--closes", default="0.75,0.9", help="fixed-grip closures to try (quote a comma list in PowerShell)"
    )
    parser.add_argument(
        "--grip",
        choices=("fixed", "force"),
        default="fixed",
        help="force: close until the fingertips feel the grip force (--closes is then ignored)",
    )
    parser.add_argument(
        "--grasp-mode",
        choices=("fingers", "weld"),
        default="fingers",
        help="weld: weld the part to the hand once the grasp holds",
    )
    parser.add_argument(
        "--thumb", type=float, default=None, help="force grip: close the thumb on its own until its tip reads this (N)"
    )
    parser.add_argument("--parts", default=None, help="part indices of the episode, e.g. 2 or 0,1,2 (default all)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "carry" / f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"  # two runs started together must not share a folder
    out.mkdir(parents=True, exist_ok=True)

    from handoff.sim.genesis_sim import GenesisSimulator

    t0 = time.time()
    sim = GenesisSimulator(str(cpu_config(out)))
    layout = Layout()
    sim.build(make_scene(layout))
    print("built", round(time.time() - t0), "s", flush=True)

    # Timing hooks on this instance: every wrist move and every settle, in sim seconds, tagged with the running skill.
    rec = {"skill": None, "moves": [], "settles": []}
    real_move, real_settle = sim.move_wrist, sim.settle

    def move_wrist(arm, pos, quat, **kw):
        dist = float(np.linalg.norm(sim.wrist_pose(arm).pos - np.asarray(pos, dtype=float)))
        planned = kw.get("duration_s") or float(
            np.clip(dist / 0.25, 0.3, 2.0)
        )  # as GenesisSimulator.move_wrist plans it
        s0, w0 = sim.time, time.time()
        err = real_move(arm, pos, quat, **kw)
        took = sim.time - s0
        rec["moves"].append(
            {
                "skill": rec["skill"],
                "dist_m": round(dist, 3),
                "planned_s": round(planned, 2),
                "took_s": round(took, 2),
                "tail_s": round(max(0.0, took - planned), 2),
                "guarded": bool(kw.get("stop_on")),
                "wall_s": round(time.time() - w0, 2),
                "err_m": round(float(err), 3),
            }
        )
        return err

    def settle(seconds):
        rec["settles"].append({"skill": rec["skill"], "s": seconds})
        return real_settle(seconds)

    real_stopped = sim._stopped_on  # noqa: SLF001

    def stopped_on(arm, guard):  # at each stop, what the contact solver (truth) says the hand touches right then
        stop = real_stopped(arm, guard)
        if stop and sim.bump_log:
            counts = sim.truth_hand_contacts(arm, guard.boxes)
            sim.bump_log[-1]["truth"] = {b: int(c) for b, c in counts.items() if c} or "nothing"
            sim.bump_log[-1]["peaks"] = {n: round(d.peak, 2) for n, d in guard.detectors.items()}
        return stop

    sim.move_wrist, sim.settle, sim._stopped_on = move_wrist, settle, stopped_on  # noqa: SLF001
    real_step = sim.step
    tilt = {"max": 0.0}

    def step(n=1):  # the hand's tilt from level every physics step: the angle of its up axis (y at the grasp) from vertical
        for _ in range(n):
            real_step(1)
            w, x, y, z = sim.wrist_pose("left").quat
            tilt["max"] = max(tilt["max"], float(np.degrees(np.arccos(np.clip(2 * (y * z + w * x), -1.0, 1.0)))))

    sim.step = step

    from score_bump_sensors import parse_seeds

    import handoff.skills as skills_module

    if args.keep_keeper:  # the A/B: the weld without stopping the keeper (as before Oct 21 2:40 PM)
        skills_module.stop_grip_keeper_after_weld = False
    runs = []
    for seed in parse_seeds(args.seeds) if args.seeds else [args.seed]:
        episode = make_episode(layout, seed)
        picks = range(len(episode.parts)) if args.parts is None else [int(v) for v in args.parts.split(",")]
        for close in [float(v) for v in args.closes.split(",")] if args.grip == "fixed" else [None]:
            for i in picks:
                part = episode.parts[i]
                target = "rework" if episode.truth(part).defective else "good"
                sim.reset()
                place_episode(sim, layout, episode)
                sim.set_object_pose(part, layout.slot_pos(0), 0.0)  # the line squares the part up in the station
                sim.set_joints(TRAVEL_POSE)
                real_settle(1.0)
                truth = episode_layout(layout, episode)
                ctx = TaskContext(
                    layout=layout,
                    target_part=part,
                    parts=episode.parts,
                    target_bin=target,
                    grasp_mode=args.grasp_mode,
                    grasp_control=args.grip,
                    grip_thumb_n=args.thumb,
                    close_fraction=close,
                    bins_seen={n: (cx, cy) for n, cx, cy in truth.bin_xy},
                )
                sim.bins_seen = ctx.bins_seen
                for name in SETUP:
                    r = SKILLS[name](sim, ctx)
                    assert r.ok, (name, r.reason)
                rec["moves"], rec["settles"] = [], []
                bumps_before = len(sim.bump_log or [])
                fallbacks_before = len(sim.line_fallback_log or [])
                skills, left_hand_in, tips, s_start, w_start = [], None, None, sim.time, time.time()
                sink = {}  # skill -> deepest hand-link penetration into the part after it, mm (contact solver: truth)
                for name in FLOW:
                    rec["skill"] = name
                    tilt["max"] = 0.0
                    r = SKILLS[name](sim, ctx)
                    q = sim.wrist_pose(ctx.arm).quat  # heading of the hand's x axis in the world xy plane, degrees
                    w, x, y, z = q
                    heading = float(np.degrees(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))))
                    skills.append(
                        {"skill": name, "ok": r.ok, "reason": r.reason, "hand_heading_deg": round(heading, 1), "max_tilt_deg": round(tilt["max"], 1)}
                    )
                    if name == "grasp":
                        tips = {k: round(v, 2) for k, v in (r.data.get("tips") or {}).items()}
                    if name in ("grasp", "lift", "move_to_bin", "lower"):
                        pen = sim.truth_finger_penetration(ctx.arm, part)
                        sink[name] = round(1000 * max(pen.values(), default=0.0), 1)
                    if name in ("lift", "move_to_bin", "lower") and left_hand_in is None:
                        if not part_in_hand(snapshot(sim, layout, (ctx.arm,), episode.parts), ctx):
                            left_hand_in = name
                    if not r.ok and name in ("approach", "grasp", "lift"):
                        break
                pos = sim.object_state(part).pos
                per_skill = defaultdict(lambda: {"planned_s": 0.0, "tail_s": 0.0, "moves": 0, "settle_s": 0.0})
                for m in rec["moves"]:
                    p = per_skill[m["skill"]]
                    p["planned_s"] += m["planned_s"]
                    p["tail_s"] += m["tail_s"]
                    p["moves"] += 1  # noqa: E702
                for s in rec["settles"]:
                    per_skill[s["skill"]]["settle_s"] += s["s"]
                run = {
                    "seed": seed,
                    "close": close,
                    "part": part,
                    "index": i,
                    "target": target,
                    "skills": skills,
                    "left_hand_in": left_hand_in,
                    "tips": tips,
                    "sink_mm": sink,
                    "part_end": [round(float(v), 3) for v in pos],
                    "sim_s": round(sim.time - s_start, 2),
                    "wall_s": round(time.time() - w_start, 1),
                    "bumps": ctx.bumps,
                    "bump_reports": (sim.bump_log or [])[bumps_before:],
                    "line_fallbacks": (sim.line_fallback_log or [])[fallbacks_before:],
                    "per_skill": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in per_skill.items()},
                    "moves": rec["moves"],
                }
                runs.append(run)
                print(
                    f"seed {seed} close {close} {part} -> {target}: "
                    + " ".join(f"{s['skill']}={'ok' if s['ok'] else 'FAIL'}" for s in skills)
                    + " | heading "
                    + " ".join(f"{s['skill']}:{s['hand_heading_deg']}" for s in skills)
                    + f" | sink mm {sink} | left hand in {left_hand_in} | tips {tips} | sim {run['sim_s']} s, wall {run['wall_s']} s",
                    flush=True,
                )
                for b in run["bump_reports"]:
                    print(
                        f"    BUMP {b.get('sensor')} stat {b.get('stat')} {b.get('unit')} (threshold {b.get('threshold')}) at step {b.get('step')}, box {b.get('box')} (est.), truth {b.get('truth', '-')}, peaks {b.get('peaks')}",
                        flush=True,
                    )
                for k, v in run["per_skill"].items():
                    print(
                        f"    {k:12s} moves {v['moves']:2d}  eased {v['planned_s']:5.2f} s  settle tails {v['tail_s']:5.2f} s  settle() {v['settle_s']:5.2f} s",
                        flush=True,
                    )
    held = sum(1 for r in runs if r["left_hand_in"] is None and all(k["ok"] for k in r["skills"]))
    print(
        f"SUMMARY {len(runs)} carries (largest hand tilt carrying {max((k['max_tilt_deg'] for r in runs for k in r['skills'] if k['skill'] in ('lift', 'move_to_bin', 'lower')), default=0)} deg): {held} with every skill ok; bumps {sum(len(r['bumps']) for r in runs)} in {sum(1 for r in runs if r['bumps'])} carries "
        f"({sum(1 for r in runs for b in r['bump_reports'] if b.get('truth') not in (None, 'nothing'))} with a truth contact at the stop); "
        f"deepest sink {max((max(r['sink_mm'].values(), default=0) for r in runs), default=0)} mm; mean cycle {np.mean([r['sim_s'] for r in runs]):.1f} s sim",
        flush=True,
    )
    (out / "result.json").write_text(
        json.dumps({"seed": args.seed, "seeds": args.seeds, "runs": runs}, indent=1), encoding="utf-8"
    )
    print(f"written {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
