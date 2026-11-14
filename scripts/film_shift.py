"""Film one capsule shift for the video (P2-A in media/film/SCRIPT.md): in-sim cameras at a fixed rate of sim time.

Runs one shift as run_capsules.py does (lean, oracle answers, the tile floor and steel table) and saves every camera's
frames at --fps of SIM time, so the clip plays smooth whatever the sim's speed; ffmpeg joins them afterwards. The
camera angles are in the work cell's own frame (the robot parked at the origin facing +x, the tray ahead), so a shot
looks the same whichever cell the seed works in.

    .\\run.ps1 scripts/film_shift.py --seed 5000 --stills --out output/film/p2a     # one still per angle, no shift
    .\\run.ps1 scripts/film_shift.py --seed 5000 --angles front,side --out output/film/p2a

Writes <out>/<angle>/fNNNNN.jpg (still.png with --stills), <out>/events.jsonl (sim time of every event the shift prints) and <out>/run.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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

LEAN = {"sensors": None, "help_text": False, "line_step_m": 0.05}

# angle -> (camera position, look-at point), cell frame: x ahead of the parked robot, y to its left, z above the table
# top. fov in degrees.
ANGLES = {
    "front": ((1.35, 0.75, 0.75), (0.20, 0.0, 0.25), 45),  # from beyond the table, a three-quarter view of the robot
    "closeup": ((0.85, 0.25, 0.45), (0.30, 0.05, 0.02), 45),  # across the table, low: the hands, the capsules, both bins
    "high": ((0.55, 0.0, 1.30), (0.32, 0.0, 0.0), 55),  # over the tray, looking down past the robot's hands
    "wide": ((2.10, -1.60, 1.20), (0.0, 0.0, 0.30), 50),  # the whole cell and the aisle, for the drive in
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second of SIM time")
    parser.add_argument("--angles", default=",".join(ANGLES), help=f"comma-separated; known: {', '.join(ANGLES)}")
    parser.add_argument("--res", default="1920x1080")
    parser.add_argument("--stills", action="store_true", help="place the episode, save one still per angle, and stop")
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "film" / "p2a"))
    args = parser.parse_args()
    angles = [a.strip() for a in args.angles.split(",") if a.strip()]
    unknown = [a for a in angles if a not in ANGLES]
    if unknown:
        parser.error(f"unknown angles {unknown}; known: {', '.join(ANGLES)}")
    w, h = (int(v) for v in args.res.lower().split("x"))

    from PIL import Image

    from handoff.sim.base import CameraSpec
    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    episode = make_capsule_episode(cells, args.seed)
    cell = cells.cell(episode.work_cell or "a")
    top = cells.layout.table_top

    def world(p: tuple[float, float, float]) -> tuple[float, float, float]:
        return (*cell.to_world(p[0], p[1]), top + p[2])

    cams = tuple(CameraSpec(f"film_{a}", pos=world(ANGLES[a][0]), lookat=world(ANGLES[a][1]), fov=ANGLES[a][2], res=(w, h)) for a in angles)
    scene = make_two_cell_scene(cells)
    sim = GenesisSimulator(args.config, **LEAN)
    sim.build(replace(scene, cameras=scene.cameras + cams))

    def shoot(name: str) -> None:
        for a in angles:
            img = Image.fromarray(sim.capture(f"film_{a}").rgb)
            img.save(out / a / name, **({"quality": 95} if name.endswith(".jpg") else {}))

    for a in angles:
        (out / a).mkdir(exist_ok=True)
    place_capsule_episode(sim, cells, episode)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    if args.stills:
        shoot("still.png")
        print(json.dumps({"seed": args.seed, "work_cell": episode.work_cell, "stills": [str(out / a / "still.png") for a in angles]}), flush=True)
        return 0

    frames = [0]
    next_t = [sim.time]
    real_step = sim.step
    t0 = sim.time

    def step(n: int = 1) -> None:
        real_step(n)
        while sim.time >= next_t[0]:  # one frame per 1/fps of sim time; a long step can owe more than one
            shoot(f"f{frames[0]:05d}.jpg")  # JPEG 95: a 1080p shift is ~2,000 frames an angle
            frames[0] += 1
            next_t[0] += 1.0 / args.fps

    sim.step = step  # type: ignore[method-assign]
    events = (out / "events.jsonl").open("w", encoding="utf-8")

    def on_event(e: str) -> None:
        events.write(json.dumps({"t": round(sim.time - t0, 3), "frame": frames[0], "event": e}) + "\n")
        events.flush()
        if not e.startswith("asked_"):
            print(f"  {sim.time - t0:6.2f}s  {e}", flush=True)

    holder: dict = {}
    ctx = CapsuleContext(cells=cells, episode=episode, block=16, logbook=Logbook(out / "log"),
                         ladder=Ladder([Step(OracleRung(lambda q: holder["ctx"].truth_now), 0.5)]), on_event=on_event)
    holder["ctx"] = ctx
    wall = time.perf_counter()
    result = run_capsules(sim, ctx)
    events.close()
    run = {"seed": args.seed, "work_cell": episode.work_cell, "status": result.status.value, "checks": result.checks,
           "fps": args.fps, "frames": frames[0], "sim_s": round(sim.time - t0, 2), "wall_s": round(time.perf_counter() - wall, 1),
           "angles": {a: {"pos": ANGLES[a][0], "lookat": ANGLES[a][1], "fov": ANGLES[a][2]} for a in angles}, "res": [w, h]}
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
