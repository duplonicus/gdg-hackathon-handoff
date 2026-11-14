"""Genesis CPU check of the bump guard inside the real skill flow: does it stay quiet when nothing is touched?

Per seed: reset, the episode's parts and trays (trays nudged up to 2 cm, ``Layout(bin_jitter_m=0.02)``), then
drive_to_table, look_at_table, reach, approach, grasp, lift, move_to_bin, lower, release and go_home with the default
bump config (config/bump.yaml: hand_imu decides, monitor_all). The robot aims at the trays' true positions
(``bins_seen`` from the episode, standing in for the bins camera). For every guarded move it records each detector's
peak and whether it fired, and, as TRUTH only, how many steps the contact solver saw the hand touch a guarded box.
Backs the "no false alarms" check of the bump guard: on Oct 21 (one CPU run, weld grasp, seeds 1 to 3) every skill
succeeded, ctx.bumps stayed empty, no detector fired on any guarded move and none touched a guarded box; the peaks
were at most 1.14 m/s^2 on the hand IMU and 0.40 N*m on the joint torque.

Runs on gs.cpu with the decimated robot and the cameras off (no GPU), about a minute plus 10 s per seed:

    .\\run.ps1 python scripts/probe_skills_bump.py --seeds 1-3
    .\\run.ps1 python scripts/probe_skills_bump.py --seeds 1-3 --grasp-mode fingers --grip force

Prints a line per skill and per guarded move, and the summary as one JSON line (also written to
output/probe/skills_bump/<timestamp>/result.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402
from score_bump_sensors import parse_seeds  # noqa: E402

from handoff import bump  # noqa: E402
from handoff.scene import Layout, episode_layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext  # noqa: E402

FLOW = ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift", "move_to_bin", "lower", "release", "go_home")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-3", help="episode seeds, e.g. 1-3 (a comma list must be quoted in PowerShell)")
    parser.add_argument("--grasp-mode", choices=("weld", "fingers"), default="weld")
    parser.add_argument("--grip", choices=("fixed", "force"), default="fixed", help="skills' grasp_control (default fixed, the skills' default)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "skills_bump" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from handoff.sim.genesis_sim import GenesisSimulator

    t0 = time.time()
    sim = GenesisSimulator(str(cpu_config(out)))
    layout = Layout(bin_jitter_m=0.02)
    sim.build(make_scene(layout))
    print("built", round(time.time() - t0), "s", flush=True)

    guards: list[tuple[bump.BumpGuard, dict]] = []
    current = {"skill": None}
    real_init, real_step = bump.BumpGuard.__init__, bump.BumpGuard.step

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self._rec = {"skill": current["skill"], "n": 0, "truth_steps": 0, "first_truth": None}
        guards.append((self, self._rec))

    def step(self):
        fired = real_step(self)
        rec = self._rec
        rec["n"] += 1
        if sum(self.sim.truth_hand_contacts(self.arm, self.boxes).values()) > 0:
            rec["truth_steps"] += 1
            if rec["first_truth"] is None:
                rec["first_truth"] = rec["n"] - 1
        return fired

    bump.BumpGuard.__init__, bump.BumpGuard.step = init, step
    seeds = []
    try:
        for seed in parse_seeds(args.seeds):
            sim.reset()
            episode = make_episode(layout, seed)
            place_episode(sim, layout, episode)
            sim.set_joints(TRAVEL_POSE)
            sim.settle(1.0)
            truth_layout = episode_layout(layout, episode)
            ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, grasp_mode=args.grasp_mode, grasp_control=args.grip,
                              bins_seen={name: (cx, cy) for name, cx, cy in truth_layout.bin_xy})
            sim.bins_seen = ctx.bins_seen
            guards.clear()
            skills = []
            for name in FLOW:
                current["skill"] = name
                r = SKILLS[name](sim, ctx)
                skills.append({"skill": name, "ok": r.ok, "reason": r.reason})
                print(f"seed {seed} {name}: ok={r.ok} {r.reason}", flush=True)
                if not r.ok and name in ("grasp", "lift"):
                    break
            moves = []
            for guard, rec in guards:
                moves.append({**rec, "peaks": {n: round(d.peak, 3) for n, d in guard.detectors.items()},
                              "fired_at": {n: d.fired_at for n, d in guard.detectors.items()}})
                print(f"  guarded move in {rec['skill']}: {rec['n']} steps, truth contact steps {rec['truth_steps']} (first {rec['first_truth']}), "
                      f"peaks {moves[-1]['peaks']}, fired_at {moves[-1]['fired_at']}", flush=True)
            seeds.append({"seed": seed, "part": ctx.target_part, "skills": skills, "bumps": ctx.bumps, "guarded_moves": moves})
    finally:
        bump.BumpGuard.__init__, bump.BumpGuard.step = real_init, real_step

    all_moves = [m for s in seeds for m in s["guarded_moves"]]
    sensors = sorted({n for m in all_moves for n in m["peaks"]})
    summary = {
        "seeds": [s["seed"] for s in seeds], "grasp_mode": args.grasp_mode, "grip": args.grip, "decider": (sim.bump_config.sensor if sim.bump_config else None),
        "all_skills_ok": all(k["ok"] for s in seeds for k in s["skills"]) and all(len(s["skills"]) == len(FLOW) for s in seeds),
        "bumps": sum(len(s["bumps"]) for s in seeds), "guarded_moves": len(all_moves),
        "moves_with_truth_contact": sum(1 for m in all_moves if m["truth_steps"] > 0),
        "fired": sorted({n for m in all_moves for n, at in m["fired_at"].items() if at is not None}),
        "max_peak": {n: max(m["peaks"].get(n, 0.0) for m in all_moves) for n in sensors},
        "wall_s": round(time.time() - t0, 1),
    }
    (out / "result.json").write_text(json.dumps({"summary": summary, "per_seed": seeds}, indent=1))
    print(json.dumps(summary))
    print("wrote", out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
