"""Where does a capsule shift's wall time go? (Nov 5; dup: "profile it and tell me the result")

Runs one shift exactly as run_capsules.py does (oracle answers), under cProfile, and prints the wall time split into
the parts that matter: the physics step, the viewer's drawing, the camera renders the questions use, IK solves, and
everything else (our Python). Then the top functions by own time. Use --config config/handoff_demo.yaml for the viewer.

    .\\run.ps1 scripts/profile_shift.py --seed 1004 --config config/handoff.yaml --out output/bench/profile_headless
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.capsule_cell import TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode  # noqa: E402
from handoff.decider import Ladder, Step  # noqa: E402
from handoff.logbook import Logbook  # noqa: E402
from handoff.rungs import OracleRung  # noqa: E402
from handoff.skills import TRAVEL_POSE  # noqa: E402
from handoff.tasks.capsules import CapsuleContext, run_capsules  # noqa: E402

# (label, substrings of "file:function" whose cumulative time counts for it); first match wins, in this order
BUCKETS = (
    ("viewer drawing", ("visualizer.py:update", "viewer.py:update", "pyrender", "rasterizer")),
    ("camera renders (questions)", ("camera.py:render", "capture_cloud", "genesis_sim.py:capture")),
    ("IK solves", ("solve_ik", "inverse_kinematics")),
    ("physics step", ("rigid_solver", "simulator.py:step", "scene.py:step")),
)


# The capsule task reads none of the robot's sensors, takes no picture with its cameras (off in capsules*.yaml) and needs
# no viewer help text; IK every 5 cm instead of 1 cm (profile_shift.py, Nov 5: these were about half the viewer run).
LEAN = {"sensors": None, "help_text": False, "line_step_m": 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"), help="capsules.yaml headless, capsules_demo.yaml with the viewer")
    parser.add_argument("--out", default=str(ROOT / "output" / "bench" / "profile"))
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    sim = GenesisSimulator(args.config, **LEAN)
    t0 = time.time()
    sim.build(make_two_cell_scene(cells))
    print(f"built in {time.time() - t0:.0f} s", flush=True)
    holder: dict = {}
    ladder = Ladder([Step(OracleRung(lambda q: holder["ctx"].truth_now), 0.5)])
    episode = make_capsule_episode(cells, args.seed)
    sim.reset()
    place_capsule_episode(sim, cells, episode)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    ctx = CapsuleContext(cells=cells, episode=episode, ladder=ladder, logbook=Logbook(out), block=16)
    holder["ctx"] = ctx
    prof = cProfile.Profile()
    sim_t0, wall0 = sim.time, time.time()
    prof.enable()
    result = run_capsules(sim, ctx)
    prof.disable()
    wall = time.time() - wall0
    sim_s = sim.time - sim_t0
    prof.dump_stats(str(out / "shift.prof"))
    print(f"shift: {result.status.value}, {sim_s:.1f} s of sim in {wall:.1f} s wall = {sim_s / wall:.2f}x real time "
          f"({sim_s / 0.01:.0f} physics steps of 0.01 s, {wall / (sim_s / 0.01) * 1000:.1f} ms wall each)", flush=True)

    stats = pstats.Stats(prof)
    entries = {f"{Path(k[0]).name}:{k[2]}": (v[2], v[3]) for k, v in stats.stats.items()}  # own time, cumulative
    claimed = {}
    for label, keys in BUCKETS:
        best = max((cum for name, (_, cum) in entries.items() if any(s in name for s in keys)), default=0.0)
        claimed[label] = best
    print("\nwhere the wall time went (cumulative, a call inside another can count in both; read as an upper bound each):")
    for label, secs in claimed.items():
        print(f"  {label:28s} {secs:7.1f} s  {secs / wall * 100:5.1f}%")
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(25)
    print("\ntop functions by own time:\n" + "\n".join(buf.getvalue().splitlines()[6:40]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
