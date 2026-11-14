"""Run sorting episodes with the ladder, logging every decision.

    bash run.sh scripts/run_sort.py --episodes 5 --ladder oracle          # no models: the sim's truth answers
    bash run.sh scripts/run_sort.py --episodes 20 --ladder local --fault drop_on_carry

Ladder configs (rungs in climbing order, each with its confidence threshold):
  oracle   : the sim's truth only (pipeline check, and labels for the log)
  local    : decider-2b with the frame -> gemma4:e4b -> oracle as the stand-in person
  offline  : the same without any cloud rung (there is none yet; kept as a named config)
  gemma    : gemma4:e4b alone -> oracle
--scales puts a load cell under each tray and makes it the first rung for in_bin (handoff/sensors.py): a settled
reading answers alone, an unclear one goes on up the ladder. Every run writes load_cells.log: the trays' true load
at each in_bin check (no device noise, no extra stepping), and with --scales the scales' tare, reading and verdict.
Thresholds are the starting values from the Jev routing convention (0.9 act, else escalate);
scripts/benchmark.py measures the real ones from the log.
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

from handoff.decider import QUESTIONS, Ladder, Step  # noqa: E402
from handoff.logbook import Logbook  # noqa: E402
from handoff.rungs import DeciderRung, GemmaRung, OracleRung  # noqa: E402
from handoff.scene import Layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.sensors import ScaleRung, TrayScales, load_cell_line, part_counts, scale_check_line, tray_grams  # noqa: E402
from handoff.skills import TRAVEL_POSE, grip_settings  # noqa: E402
from handoff.tasks.sort import SortContext, run_sort, truth  # noqa: E402


def build_ladder(name: str, sim, ctx_holder: dict, thresholds: dict[str, float], scales: bool = False) -> Ladder:
    def oracle_truth(question):
        ctx = ctx_holder["ctx"]
        return truth(sim, ctx, ctx_holder["part"], question.id)

    oracle = OracleRung(oracle_truth, name="oracle-human")
    steps = []
    if scales:
        steps.append(Step(ScaleRung(lambda: ctx_holder["ctx"].scale_now), 1.0, frozenset({"in_bin"})))
    if name == "oracle":
        return Ladder([*steps, Step(oracle, 0.0)])
    if name in ("local", "offline"):
        steps.append(Step(DeciderRung("decider-2b", with_image=True), thresholds.get("decider-2b", 0.9)))
        steps.append(Step(GemmaRung("gemma4:e4b"), thresholds.get("gemma4:e4b", 0.9)))
    elif name == "gemma":
        steps.append(Step(GemmaRung("gemma4:e4b"), thresholds.get("gemma4:e4b", 0.9)))
    else:
        raise SystemExit(f"unknown ladder {name}")
    steps.append(Step(oracle, 0.0))
    return Ladder(steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=100)
    parser.add_argument("--ladder", default="oracle", choices=("oracle", "local", "offline", "gemma"))
    parser.add_argument("--threshold", action="append", default=[], help="rung=value, e.g. decider-2b=0.8")
    parser.add_argument("--grasp", default="fingers", choices=("fingers", "weld"))
    parser.add_argument("--grip", default="fixed", choices=("fixed", "force", "touch"), help="how the hand closes: fixed closure, fingertip force with the grip keeper, or touch (fingertips feel the part, then weld; implies --grasp weld). Stays fixed by default so the PREREG test03/test04 commands rerun as recorded; the demo defaults to touch")
    parser.add_argument("--close", type=float, default=None, help="fixed grip: the closure to close to (default 0.9)")
    parser.add_argument("--fault", default=None, choices=(None, "drop_on_carry", "crowded", "wrong_bin", "cycle"), help="cycle = none, drop_on_carry, crowded in turn")
    parser.add_argument("--parts", type=int, default=None, help="parts per episode (default: all slots)")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--bin-jitter", type=float, default=0.02, help="move each bin up to this far (m, each axis) per episode; the robot has to find them")
    parser.add_argument("--no-return-home", action="store_true", help="after a release, lift clear and start back without waiting for the arm to settle at home")
    parser.add_argument("--dance", action="store_true", help="the macarena after a verified sort (off until dup is happy with everything; it backs away from the table first)")
    parser.add_argument("--scales", action="store_true", help="a load cell under each tray answers in_bin first (handoff/sensors.py)")
    parser.add_argument("--frames", action="store_true", help="also save event frames from the head and close-up cameras")
    args = parser.parse_args(argv)
    if args.grip != "fixed" and args.close is not None:
        parser.error(f"--close sets the fixed grasp; it does nothing with --grip {args.grip}")
    return args


def main() -> int:
    args = parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    thresholds = dict(kv.split("=") for kv in args.threshold)
    thresholds = {k: float(v) for k, v in thresholds.items()}
    out = Path(args.out) if args.out else ROOT / "output" / "sort" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout(bin_jitter_m=args.bin_jitter)
    sim = GenesisSimulator(args.config)
    t0 = time.time()
    sim.build(make_scene(layout))
    print(f"built in {time.time() - t0:.1f} s, results in {out}")
    holder: dict = {}
    ladder = build_ladder(args.ladder, sim, holder, thresholds, scales=args.scales)
    logbook = Logbook(out)
    part_grams = TrayScales(layout).part_grams
    summary = []
    # Resumable: a run that died keeps its finished episodes; the same command skips them (episodes.jsonl is the record).
    done_seeds = {json.loads(line)["seed"] for line in (out / "episodes.jsonl").open(encoding="utf-8")} if (out / "episodes.jsonl").exists() else set()
    if done_seeds:
        print(f"resuming: {len(done_seeds)} episodes already in {out}")
    for k in range(args.episodes):
        seed = args.seed0 + k
        if seed in done_seeds:
            continue
        sim.reset()
        episode = make_episode(layout, seed, n_parts=args.parts)
        place_episode(sim, layout, episode)
        sim.set_joints(TRAVEL_POSE)
        sim.settle(0.5)
        logbook.episode = f"ep{seed:04d}"

        def save_frame(tag: str, prefix: str = f"ep{seed:04d}") -> None:
            if not args.frames:
                return
            for cam in ("head", "debug"):
                frame = sim.capture(cam)
                if frame.rgb is not None:
                    Image.fromarray(frame.rgb).save(out / "frames" / f"{prefix}_{tag}_{cam}.png")

        fault = (None, "drop_on_carry", "crowded")[k % 3] if args.fault == "cycle" else args.fault
        ctx = SortContext(layout=layout, episode=episode, ladder=ladder, logbook=logbook, **{"grasp_mode": args.grasp, **grip_settings(args.grip)}, close_fraction=args.close, fault=fault, on_event=save_frame, return_home=not args.no_return_home, dance=args.dance,
                          scales=TrayScales(layout, seed=seed) if args.scales else None)
        holder["ctx"] = ctx

        # the oracle needs to know which part a question is about: the tree sets it through the context
        original_ask_part = {"part": None}

        import handoff.tasks.sort as sort_module

        real_ask = sort_module.ask

        counts: dict = dict.fromkeys(layout.bin_names, 0)  # parts per tray at the last in_bin check (the trays start empty)

        def tracking_ask(sim_, ctx_, qid, part, _seed=seed, _counts=counts):
            holder["part"] = part
            original_ask_part["part"] = part
            d = real_ask(sim_, ctx_, qid, part)
            if qid == "in_bin":
                grams = tray_grams(sim_, layout.bin_names)
                lines = [load_cell_line(grams, part_grams, _counts)]
                if ctx_.scale_now is not None:
                    lines.append(scale_check_line(part, ctx_.scale_now))
                _counts.clear()
                _counts.update(part_counts(grams, part_grams))
                with (out / "load_cells.log").open("a", encoding="utf-8") as f:
                    f.writelines(f"ep{_seed:04d} {part}: {line}\n" for line in lines)
            return d

        sort_module.ask = tracking_ask
        wall0 = time.time()
        fallbacks_before = len(getattr(sim, "line_fallback_log", None) or [])
        try:
            result = run_sort(sim, ctx)
        finally:
            sort_module.ask = real_ask
        row = {
            "seed": seed,
            "ladder": args.ladder,
            "scales": args.scales,
            "grasp": ctx.grasp_mode,  # what the hand did: --grip touch welds whatever --grasp says
            "grip": args.grip,
            "fault": fault,
            "parts": list(episode.parts),
            "finishes": {p: episode.truth(p).label for p in episode.parts},
            "status": result.status.value,
            "checks": result.checks,
            "n_decisions": len(result.decisions),
            "answered_by": {d.question_id + ":" + d.answered_by: 1 for d in result.decisions},
            "wrong": [(d.question_id, d.answer, d.truth, d.answered_by) for d in result.decisions if d.truth is not None and d.answer != d.truth],
            "bins_true": {name: [round(cx, 4), round(cy, 4)] for name, cx, cy in ctx.layout.bin_xy},
            "bins_seen": {name: [round(x, 4), round(y, 4)] for name, (x, y) in ctx.bins_seen.items()},
            "sim_time": round(result.sim_time, 1),
            "wall_s": round(time.time() - wall0, 1),
            "line_fallbacks": (getattr(sim, "line_fallback_log", None) or [])[fallbacks_before:],  # IK branch flips
        }
        summary.append(row)
        (out / "episodes.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        (out / f"trace_ep{seed:04d}.txt").write_text(result.trace.text(), encoding="utf-8")
        c = result.checks
        print(
            f"episode {seed}: {result.status.value:7s} correct bin {c['parts_in_correct_bin']}/{c['n_parts']}, in any bin {c['parts_in_any_bin']}, "
            f"danced {c['danced']}, decisions {len(result.decisions)}, wrong {len(row['wrong'])}  (sim {row['sim_time']} s, wall {row['wall_s']} s)"
        )
    logbook.close()
    ok = sum(1 for r in summary if r["status"] == "success")
    print(f"\n{ok}/{len(summary)} episodes fully succeeded; decisions logged to {out / 'decisions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
