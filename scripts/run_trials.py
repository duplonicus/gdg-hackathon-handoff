"""Run N headless pick-and-place trials and print the success count.

    bash run.sh scripts/run_trials.py --trials 10 --grasp both

Each trial uses seed k (fixed, so a run is repeatable) to jitter the part start poses, then runs
the stage 1 task. Results go to ``output/trials/<run name>/``: one JSON line per trial in
``trials.jsonl`` and head-camera PNGs at the key moments. ``output/`` is gitignored by the kit.
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

from handoff.scene import Layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.skills import TRAVEL_POSE, TaskContext  # noqa: E402
from handoff.tasks.pick_place import run_pick_place  # noqa: E402


def parse_triple(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="first seed; trial k uses seed0 + k")
    parser.add_argument("--grasp", choices=("fingers", "weld", "both"), default="fingers")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=None, help="output folder (default output/trials/<timestamp>)")
    parser.add_argument("--grasp-offset", type=parse_triple, default=None, help="x,y,z wrist offset from the part")
    parser.add_argument("--close", type=float, default=None, help="hand close fraction 0..1")
    parser.add_argument("--squeeze", type=float, default=None, help="finger torque after closing (N*m)")
    parser.add_argument("--grip", default=None, choices=("fixed", "force"), help="how the hand closes (default fixed; --close and --squeeze need fixed)")
    parser.add_argument("--part-size", type=parse_triple, default=None, help="x,y,z part size in metres")
    parser.add_argument("--hand-kp", type=float, default=None, help="finger PD stiffness (kit default 350)")
    parser.add_argument("--hand-kv", type=float, default=None, help="finger PD damping (default kp/10)")
    parser.add_argument("--no-align", action="store_true", help="keep the hand at its home yaw instead of matching the part")
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args()
    if args.grip == "force" and (args.close is not None or args.squeeze is not None):
        parser.error("--close and --squeeze set the fixed grasp; they do nothing with --grip force")

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out) if args.out else ROOT / "output" / "trials" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout(part_size=args.part_size) if args.part_size else Layout()
    sim = GenesisSimulator(args.config)
    t0 = time.time()
    sim.build(make_scene(layout))
    if args.hand_kp is not None:
        sim.set_hand_gains("left", args.hand_kp, args.hand_kv if args.hand_kv is not None else args.hand_kp / 10.0)
    print(f"built in {time.time() - t0:.1f} s, backend {sim.kit.compute_backend}, results in {out}")

    modes = ("fingers", "weld") if args.grasp == "both" else (args.grasp,)
    summary: dict[str, list[dict]] = {}
    with (out / "trials.jsonl").open("a", encoding="utf-8") as log:
        for mode in modes:
            summary[mode] = []
            for k in range(args.trials):
                seed = args.seed0 + k
                sim.reset()
                episode = make_episode(layout, seed)
                place_episode(sim, layout, episode)
                sim.set_joints(TRAVEL_POSE)
                sim.settle(0.5)

                tag_prefix = f"{mode}_seed{seed:02d}"

                def save_frame(tag: str, prefix: str = tag_prefix) -> None:
                    if args.no_frames:
                        return
                    for cam in ("head", "debug", "inspect"):
                        frame = sim.capture(cam)
                        if frame.rgb is not None:
                            Image.fromarray(frame.rgb).save(out / f"{prefix}_{tag}_{cam}.png")

                ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, grasp_mode=mode, on_event=save_frame)
                if args.grasp_offset:
                    ctx.grasp_offset = args.grasp_offset
                if args.grip is not None:
                    ctx.grasp_control = args.grip
                if args.close is not None:
                    ctx.close_fraction = args.close
                if args.squeeze is not None:
                    ctx.squeeze_effort = args.squeeze
                if args.no_align:
                    ctx.align_yaw = False
                wall0 = time.time()
                result = run_pick_place(sim, ctx)
                record = {"mode": mode, "seed": seed, "part": episode.parts[0], "finish": episode.truth(episode.parts[0]).label, "wall_s": round(time.time() - wall0, 1), **result.as_dict()}
                log.write(json.dumps(record) + "\n")
                log.flush()
                summary[mode].append(record)
                status = "ok  " if result.success else f"FAIL at {result.failed_at or 'final check'}"
                reason = next((s["reason"] for s in record["steps"] if not s["ok"]), "")
                print(f"[{mode}] seed {seed:2d}: {status} {reason}  (sim {result.sim_time:.1f} s, wall {record['wall_s']} s)")

    print()
    for mode, records in summary.items():
        n_ok = sum(1 for r in records if r["success"])
        print(f"{mode}: {n_ok}/{len(records)} succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
