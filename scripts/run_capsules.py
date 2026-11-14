"""Run capsule shifts in Genesis and log every decision with its picture and the truth (Nov 4).

The default ladder is the simulator's truth answering every question (``oracle``), which is how a decision log is
made for replaying models on afterwards, as test03 and test04 were for the sorting task. Resumable: run the same
command again and finished episodes are skipped.

    .\\run.ps1 scripts/run_capsules.py --episodes 4 --seed0 0 --out output/capsules/smoke
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

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode  # noqa: E402
from handoff.decider import Ladder, Step  # noqa: E402
from handoff.logbook import Logbook  # noqa: E402
from handoff.rungs import OracleRung  # noqa: E402
from handoff.skills import TRAVEL_POSE  # noqa: E402
from handoff.tasks.capsules import BLOCK, CapsuleContext, run_capsules  # noqa: E402


# The capsule task reads none of the robot's sensors, takes no picture with its cameras (off in capsules*.yaml) and needs
# no viewer help text; IK every 5 cm instead of 1 cm (profile_shift.py, Nov 5: these were about half the viewer run).
LEAN = {"sensors": None, "help_text": False, "line_step_m": 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--block", type=int, default=BLOCK, help="capsules per leak_block question")
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"), help="capsules.yaml headless, capsules_demo.yaml with the viewer")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out) if args.out else ROOT / "output" / "capsules" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    sim = GenesisSimulator(args.config, **LEAN)
    import handoff.capsule_skills as capsule_skills

    capsule_skills.LOOK_AT_WORK = bool(sim._config.show_viewer)  # the head follows the work in viewer runs only (FREEZE)
    t0 = time.time()
    sim.build(make_two_cell_scene(cells))
    print(f"built in {time.time() - t0:.0f} s, results in {out}", flush=True)
    logbook = Logbook(out)
    holder: dict = {}
    ladder = Ladder([Step(OracleRung(lambda q: holder["ctx"].truth_now), 0.5)])
    rows_path = out / "episodes.jsonl"
    done = {json.loads(line)["seed"] for line in rows_path.open(encoding="utf-8")} if rows_path.exists() else set()
    if done:
        print(f"resuming: {len(done)} episodes already in {out}", flush=True)
    for k in range(args.episodes):
        seed = args.seed0 + k
        if seed in done:
            continue
        sim.reset()
        episode = make_capsule_episode(cells, seed)
        place_capsule_episode(sim, cells, episode)
        sim.set_joints(TRAVEL_POSE)
        sim.settle(0.5)
        logbook.episode = f"ep{seed:04d}"
        ctx = CapsuleContext(cells=cells, episode=episode, ladder=ladder, logbook=logbook, block=args.block,
                             on_event=lambda e: print(f"  {e}", flush=True) if not e.startswith("asked_") else None)
        holder["ctx"] = ctx
        wall0 = time.time()
        result = run_capsules(sim, ctx)
        row = {
            "seed": seed,
            "work_cell": episode.work_cell,
            "leakers": list(episode.leakers),
            "pool_sizes": list(episode.pool_sizes),
            "block": args.block,
            "status": result.status.value,
            "checks": result.checks,
            "n_decisions": len(result.decisions),
            "asked": {q: sum(1 for d in result.decisions if d.question_id == q) for q in {d.question_id for d in result.decisions}},
            "wrong": [(d.question_id, d.answer, d.truth, d.answered_by) for d in result.decisions if d.truth is not None and d.answer != d.truth],
            "sim_time": round(result.sim_time, 1),
            "wall_s": round(time.time() - wall0, 1),
        }
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    logbook.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
