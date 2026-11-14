"""The watchable demo: viewer, the kit's control panel, the operator page, the real ladder.

    .\run.ps1 scripts/demo.py                       # the frozen ladder (d3 27B, then you), a forced drop on the first part
    .\run.ps1 scripts/demo.py --scales --fault wrong_bin   # tray load cells catch a part set down in the wrong tray
    .\run.ps1 scripts/demo.py --fault crowded       # the station gets crowded: the safety rule holds the robot
    .\run.ps1 scripts/demo.py --ladder local        # the Oct 11 ladder: Decider 2B, then Gemma (in_bin only), then you
    .\run.ps1 scripts/demo.py --ladder oracle       # no models, just the choreography

The frozen ladder is the one docs/PREREG.md fixed and test04 confirmed: d3 27B answers alone at 0.80 confidence or
more, anything less goes to you. It needs the GPU to itself: the demo stops its own Decider 2B and unloads Ollama's
models before it starts d3 27B (about 18 GB of GPU memory, 27 GB of commit; it refuses past the 110 GB cap).
--scales adds the tray load cells ahead of d3 on in_bin, and --fault wrong_bin is a case neither test log holds: neither
is part of what test04 confirmed. The load cells show on the control panel (a "Tray load cells" window) and in
run.log on every run; without --scales they only show, they do not answer.

The robot drives to the table, inspects each part, sorts good ones to the green bin and
defective ones to the orange rework bin, survives the forced failure, verifies the
sort and only then dances. Questions the models are not sure about appear on the operator page;
the robot holds until you answer (or the timeout passes and it takes the safe option).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.bump import BumpConfig, accelerometer_rows, bump_line, tactile_rows, torque_rows  # noqa: E402
from handoff.sim.base import line_fallback_line
from handoff.decider import Ladder, Step  # noqa: E402
from handoff.logbook import Logbook  # noqa: E402
from handoff.operator import HumanRung, OperatorPage  # noqa: E402
from handoff.rungs import DeciderRung, GemmaRung, OracleRung, OLLAMA_URL, _get, unload_ollama_models  # noqa: E402
from handoff.servers import D3_URL, ensure_d3, ensure_decider, stop_d3, stop_decider  # noqa: E402
from handoff.voice import voice_for_demo  # noqa: E402
from handoff.scene import Layout, make_episode, make_scene, place_episode  # noqa: E402
from handoff.sensors import LoadCellWatch, ScaleRung, TrayScales, scale_check_line, scale_check_row, sensor_window_hook, tray_grams  # noqa: E402
from handoff.skills import SKILLS, TOUCH_THUMB_N, TRAVEL_POSE, grip_keeper, grip_line, grip_rows, grip_settings, grip_target_n, keeper_line  # noqa: E402
from handoff.status import RunStatus, bump_rows, finish_line, rung_text  # noqa: E402
from handoff.tasks.sort import SortContext, run_sort, truth  # noqa: E402


# The frozen ladder, copied from docs/PREREG.md (output/sort/test03/frozen-ladder.json is gitignored; checked below when present)
FROZEN_ARM, FROZEN_THRESHOLD = "d3-27b-vision", 0.80
NATIVE_OLLAMA_URL = "http://127.0.0.1:11435"


def check_frozen_file() -> None:
    path = ROOT / "output" / "sort" / "test03" / "frozen-ladder.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("rungs") != [FROZEN_ARM] or frozen.get("thresholds", {}).get(FROZEN_ARM) != FROZEN_THRESHOLD:
            raise SystemExit(f"{path} disagrees with the demo's frozen ladder ({FROZEN_ARM} at {FROZEN_THRESHOLD}); fix one of them")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ladder", default="frozen", choices=("frozen", "local", "oracle"))
    parser.add_argument("--fault", default="drop_on_carry", choices=("none", "drop_on_carry", "crowded", "wrong_bin"))
    parser.add_argument("--decider-threshold", type=float, default=None, help="one number for every question (default: the measured per-question thresholds)")
    parser.add_argument("--gemma-threshold", type=float, default=None, help="same for Gemma (default: measured; Gemma is never trusted on finish)")
    parser.add_argument("--operator-timeout", type=float, default=120.0)
    parser.add_argument("--allow-missing-models", action="store_true", help="run the local ladder even if a model server is down (every question it would have answered goes to the operator)")
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff_demo.yaml"))
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--dance", action="store_true", help="the macarena after a verified sort (off until dup is happy with everything; it backs away from the table first)")
    parser.add_argument("--scales", action="store_true", help="a load cell under each tray answers in_bin first; the models and you only when it cannot tell")
    parser.add_argument("--grip", default=None, choices=("fixed", "force", "touch"), help="how the hand closes (default touch, dup's pick Oct 27; fixed when only --close is given): touch, the fingertips feel the part, then weld; fixed, the fixed closure; force, until the fingertips feel the grip force, held by the grip keeper")
    parser.add_argument("--close", type=float, default=None, help="fixed grip: the closure to close to (default 0.9; 0.85 sinks less and carried every part tried; 0.75 drops parts on the carry, VALIDATION Oct 21)")
    parser.add_argument("--film", default=None, help="save film frames from in-sim cameras into this NEW folder (media/film/SCRIPT.md); no browser opens for the page, and the demo exits after the sort instead of holding the viewer open")
    parser.add_argument("--film-angles", default="front,wide,trays", help="comma-separated, from handoff.film.SORT_ANGLES (quote the list in PowerShell)")
    parser.add_argument("--film-fps", type=float, default=30.0, help="frames per second of SIM time")
    parser.add_argument("--film-res", default="1920x1080")
    args = parser.parse_args(argv)
    local_only = [f for f, v in (("--decider-threshold", args.decider_threshold), ("--gemma-threshold", args.gemma_threshold), ("--allow-missing-models", args.allow_missing_models or None)) if v is not None]
    if local_only and args.ladder != "local":
        parser.error(f"{', '.join(local_only)}: these apply only to --ladder local")
    if args.grip is None:  # the demo's default is the touch grip; a bare --close still means the fixed grasp it sets
        args.grip = "fixed" if args.close is not None else "touch"
    if args.grip != "fixed" and args.close is not None:
        parser.error(f"--close sets the fixed grasp; it does nothing with --grip {args.grip}")
    return args


def speech_reader(ladder: str) -> DeciderRung | None:
    """What reads a spoken answer: the server this ladder already has loaded, so no second model on the GPU. Always its
    own rung object, because the ladder's rungs get wrapped to log and track what the robot is asking."""
    if ladder == "frozen":
        return DeciderRung(FROZEN_ARM, D3_URL, with_image=True, data_url=True, timeout=120.0, expect=("d3-27b",))
    return DeciderRung("decider-2b") if ladder == "local" else None


def main() -> int:
    args = parse_args()
    if args.film:  # a used folder is refused before anything loads (d3 takes minutes and clears the GPU)
        from handoff.film import FilmRecorder

        if FilmRecorder.holds_take(Path(args.film), [n.strip() for n in args.film_angles.split(",") if n.strip()]):
            raise SystemExit(f"NOT RUNNING: {args.film} already holds a take; film into a new folder")

    d3 = None
    if args.ladder == "frozen":
        check_frozen_file()
        d3 = DeciderRung(FROZEN_ARM, D3_URL, with_image=True, data_url=True, timeout=120.0, expect=("d3-27b",))
        # Make room on the GPU every time, even when d3 is already up: d3 needs about 18 of its 24 GB, and a Decider 2B
        # (4.8 GB) or a Gemma (5 GB) left from an earlier run would push the Genesis build out of memory.
        if not stop_decider():
            raise SystemExit("NOT RUNNING: the Decider 2B on port 8765 would not stop, and d3 27B will not fit beside it")
        for url in (OLLAMA_URL, NATIVE_OLLAMA_URL):
            unload_ollama_models(url)
        try:
            ensure_d3()
            d3.check_model()  # port 8774 is shared with other 27B arms: refuse to act on the wrong one
        except RuntimeError as error:
            raise SystemExit(f"NOT RUNNING: {error}") from None
    elif args.ladder == "local":
        if not stop_d3():  # d3 27B left from a frozen run holds about 18 GB: the Decider and Gemma would not fit beside it
            raise SystemExit("NOT RUNNING: d3 27B on port 8774 would not stop; the local ladder's models will not fit beside it")
        # A model server that is down looks exactly like a model that is unsure: every question climbs to the
        # operator. dup's first real-ladder run on Windows (Oct 11) had the Decider down for all 11 decisions and
        # nothing said so. Check before the 40 s build. A Decider that is down is started here (servers.py) and
        # waited for; Ollama is a container, not ours to start.
        down = []
        try:
            ensure_decider()
        except RuntimeError as error:
            down.append(str(error))
        try:
            _get(f"{OLLAMA_URL}/api/version")
        except Exception:  # noqa: BLE001
            down.append(f"Ollama at {OLLAMA_URL} (the open-webui container)")
        if down:
            print("NOT RUNNING: " + "; and ".join(down))
            if not args.allow_missing_models:
                raise SystemExit("Fix it and run again, or use --ladder oracle (no models), or --allow-missing-models to go ahead anyway.")
            print("Going ahead: every question that model would have answered goes to the operator page.")

    from mh3_genesis import MH3ControlPanel

    from handoff.sim.genesis_sim import GenesisSimulator

    layout = Layout(bin_jitter_m=0.02)  # the bins are nudged every run; the robot finds them from the bins camera
    sim = GenesisSimulator(args.config)
    print("building the scene (about 30 s)...")
    film = None
    scene = make_scene(layout)
    if args.film:
        from dataclasses import replace

        from handoff.film import SORT_ANGLES, FilmRecorder, camera_specs

        film_names = [n.strip() for n in args.film_angles.split(",") if n.strip()]
        film_res = tuple(int(v) for v in args.film_res.lower().split("x"))
        scene = replace(scene, cameras=scene.cameras + camera_specs(SORT_ANGLES, film_names, film_res))
    sim.build(scene)
    if args.film:
        film = FilmRecorder(sim, Path(args.film), film_names, fps=args.film_fps)
    panel = None
    if not args.no_panel:
        panel = MH3ControlPanel(sim.kit)
        panel.start(open_browser=not args.film)
        sim.kit.add_step_hook(panel.update)
    page = OperatorPage(port=8790, open_browser=not args.film, voice=voice_for_demo(speech_reader(args.ladder)))
    page.start()
    print(f"operator page: {page.url}")

    episode = make_episode(layout, args.seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    sim.settle(1.0)
    holder: dict = {}
    out = ROOT / "output" / "demo" / time.strftime("%Y%m%d-%H%M%S")
    logbook = Logbook(out, f"demo{args.seed}")

    def oracle_truth(question):
        return truth(sim, holder["ctx"], holder["part"], question.id)

    if args.film:  # the scripted operator answers as someone who can see the scene (film_answerer.py, plan "truth")
        page.truth_fn = oracle_truth

    status = RunStatus(args.ladder, out / "run.log", title=f"{args.ladder} ladder" + (" + tray scales (scales not part of the confirmed ladder)" if args.scales else ""))
    status.start()
    if panel is not None:
        panel.extra_state = status.snapshot
    viewer = getattr(getattr(sim.kit.scene, "viewer", None), "_pyrender_viewer", None)
    if viewer is not None:
        # the models' state at top right and the newest log lines at bottom left (Genesis keeps its help text top left)
        try:
            from handoff import viewer_text

            viewer_text.install(viewer, lambda: status.caption_lines(2), log=status.log)
        except Exception as error:  # noqa: BLE001  (text on the viewer must never stop the run)
            status.log(f"viewer text off: {type(error).__name__}: {error}")

    # The tray load cells, always: the cells are there whether or not --scales lets them answer in_bin. A cheap read
    # (one contact force per tray, no stepping) every 10 steps, with the display's own device noise so the in_bin
    # scales' noise stream is the same as without the display; a log line each time a tray's part count settles anew.
    display_scales = TrayScales(layout, seed=args.seed + 1)
    watch = LoadCellWatch(display_scales.part_grams, answering=args.scales)

    cells = {"steps": 0, "broken": False}

    def load_cells(_simulation) -> None:
        cells["steps"] += 1
        if cells["steps"] % 10 or cells["broken"]:
            return
        try:
            grams = display_scales.noisy(tray_grams(sim, layout.bin_names))
        except Exception as error:  # noqa: BLE001  (a display must never stop the run)
            cells["broken"] = True
            status.log(f"load cells: display off, the tray read failed ({type(error).__name__}: {error})")
            return
        rows, line = watch.update(grams)
        status.set_sensor("Tray load cells", rows)
        if line:
            status.log(line)

    sim.kit.add_step_hook(load_cells)

    # The grip: a "Grip force" window with each fingertip's live force against the grip target, from the same cheap
    # read (no stepping) every 10 steps, and one log line per grasp (how it closed, every tip, held or why not).
    grip_target = grip_target_n(layout, sim.sensors.fingertip)
    grip_display = {"steps": 0, "broken": False}

    def grip_window(_simulation) -> None:
        grip_display["steps"] += 1
        if grip_display["steps"] % 10 or grip_display["broken"]:
            return
        try:
            tips = sim.read_fingertips("left")
            if tips:
                status.set_sensor("Grip force", grip_rows(tips, grip_target, sim.hand_closure("left"), keeper=grip_keeper(sim, "left")))
        except Exception as error:  # noqa: BLE001  (a display must never stop the run)
            grip_display["broken"] = True
            status.log(f"grip force: display off, the fingertip read failed ({type(error).__name__}: {error})")

    sim.kit.add_step_hook(grip_window)
    real_grasp = SKILLS["grasp"]

    def logged_grasp(sim_, tctx):
        result = real_grasp(sim_, tctx)
        status.log(grip_line(tctx.target_part, result, tctx.arm))
        return result

    SKILLS["grasp"] = logged_grasp

    # Every other sensor the robot carries gets its own window too (the IMUs, the arm's joint torques, the tactile
    # skin); the fingertips are already in "Grip force".
    sim.kit.add_step_hook(sensor_window_hook(sim, status, every=10, skip=("fingertips",)))

    # Bump detection (handoff.bump, config/bump.yaml): an "Accelerometer" and a "Joint torque" window with each
    # detector's residual against its threshold during the guarded moves, from reads only, every 10 steps; and one log
    # line per bump the robot felt (which sensor, by how much, and the box it ESTIMATES: the detectors cannot tell
    # which box, so it is the nearest in the drawing, the trays where the bins camera saw them). Also one line per
    # straight-line move that gave up for the joint path (an IK branch flip), so a wide arc on screen has its reason.
    sim.bump_config = BumpConfig.from_yaml()
    bump_display = {"steps": 0, "logged": 0, "fallbacks": 0, "broken": False}

    def bump_windows(_simulation) -> None:
        bump_display["steps"] += 1
        if bump_display["broken"]:
            return
        try:
            if "ctx" in holder:
                sim.bins_seen = holder["ctx"].bins_seen  # the robot's own picture of the trays (filled by locate-bins)
            felt = sim.bump_log or []
            for info in felt[bump_display["logged"]:]:
                status.log(bump_line(info))
            bump_display["logged"] = len(felt)
            gave_up = sim.line_fallback_log or []
            for info in gave_up[bump_display["fallbacks"]:]:
                status.log(line_fallback_line(info))
            bump_display["fallbacks"] = len(gave_up)
            if bump_display["steps"] % 10 == 0:
                status.set_sensor("Accelerometer", accelerometer_rows(sim.bump_guard))
                status.set_sensor("Joint torque", torque_rows(sim.bump_guard))
                status.set_sensor("Skin (bump)", tactile_rows(sim.bump_guard))
        except Exception as error:  # noqa: BLE001  (a display must never stop the run)
            bump_display["broken"] = True
            status.log(f"bump windows: display off ({type(error).__name__}: {error})")

    sim.kit.add_step_hook(bump_windows)
    status.log(f"bump detection: {sim.bump_config.sensor} stops the arm (config/bump.yaml); the box it names is an estimate")
    grip_how = {"force": " (closes until the fingertips feel the grip force; the grip keeper holds it)",
                "touch": f" (closes until the fingertips feel the part, the thumb to {TOUCH_THUMB_N} N, then welds: the fingers stop at the surface)"}
    status.log(f"grip: {args.grip}" + grip_how.get(args.grip, f" (closes to {args.close if args.close is not None else 0.9:.2f})"))
    status.log(f"run started: seed {args.seed}, fault {args.fault}, parts " + ", ".join(f"{p} ({episode.truth(p).label})" for p in episode.parts))

    # what the models said on THIS question, filled as each rung answers (announce below), shown on the operator page
    human = HumanRung(page, idle=lambda: sim.step(5), timeout_s=args.operator_timeout, climbed=lambda: ", ".join(holder.get("climbing", [])))
    if args.ladder == "oracle":
        ladder = Ladder([Step(OracleRung(oracle_truth, "oracle"), 0.0)])
    elif args.ladder == "frozen":
        ladder = Ladder([Step(d3, FROZEN_THRESHOLD), Step(human, 0.0)])
    else:
        # Measured Oct 11 on the 24-episode log and verified on a fresh 12-episode set (benchmark.json).
        decider_t = args.decider_threshold if args.decider_threshold is not None else {"finish": 0.85, "in_bin": 0.7, "safe_to_grasp": 0.85, "sort_complete": 0.75}
        gemma_t = args.gemma_threshold if args.gemma_threshold is not None else {"finish": 1.0, "in_bin": 0.8, "safe_to_grasp": 1.0, "sort_complete": 1.0}
        # Gemma only where it can ever be trusted (threshold under 1.0): asking it the rest only added its wait, 102 s
        # of cold load and prompt on the first question of the Oct 21 run, before the question reached the page.
        gemma_q = frozenset(q for q, t in gemma_t.items() if t < 1.0) if isinstance(gemma_t, dict) else None
        ladder = Ladder([Step(DeciderRung("decider-2b"), decider_t), Step(GemmaRung("gemma4:e4b"), gemma_t, gemma_q), Step(human, 0.0)])
    if args.scales:
        ladder.steps.insert(0, Step(ScaleRung(lambda: holder["ctx"].scale_now), 1.0, frozenset({"in_bin"})))

    def announce(rung) -> None:
        """Say who is being asked before each call, so a model thinking (or the page waiting) never looks like a hang."""
        real = rung.answer

        def answer(question, state_text, image_b64):
            part = holder.get("part", "")
            if rung.name == "scale" and holder["ctx"].scale_now is not None:  # the reading, before anyone else is asked
                status.log(scale_check_line(part, holder["ctx"].scale_now))
                watch.extra_rows = [scale_check_row(part, holder["ctx"].scale_now)]
            elif rung.name == "human":
                status.log(f"{question.id} {part}: waiting for you on the operator page {page.url} (safe option '{question.safe_option}' after {args.operator_timeout:.0f} s)")
            elif rung.name not in ("scale", "oracle"):
                status.log(f"{question.id} {part}: asking {rung.name}...")
            ans = real(question, state_text, image_b64)
            if rung.name != "human":
                holder.setdefault("climbing", []).append(rung_text(ans))
            return ans

        rung.answer = answer

    for step in ladder.steps:
        announce(step.rung)

    import handoff.tasks.sort as sort_module

    real_ask = sort_module.ask

    def tracking_ask(sim_, ctx_, qid, part):
        holder["part"] = part
        holder["climbing"] = []
        d = real_ask(sim_, ctx_, qid, part)
        if not holder.get("bins_logged") and ctx_.bins_seen:
            holder["bins_logged"] = True
            status.log("bins found at " + ", ".join(f"{name} ({x:.3f}, {y:.3f})" for name, (x, y) in ctx_.bins_seen.items()))
        status.decision(d)
        print(f"  [{qid}] -> {d.answer} by {d.answered_by}  (truth {d.truth})  " + "; ".join(f"{r.rung} {r.top} {r.confidence:.2f} {r.latency_ms:.0f} ms" for r in d.rungs))
        return d

    sort_module.ask = tracking_ask

    pace = {"sim": sim.time, "wall": time.perf_counter()}

    def paced(tag: str) -> str:
        """The event with the sim clock and how fast the sim ran since the last event (sim s per wall s): the demo
        looked slow on Oct 21 (about 0.4 x real time from the bump stamps), and this says where."""
        now_sim, now_wall = sim.time, time.perf_counter()
        rate = (now_sim - pace["sim"]) / max(now_wall - pace["wall"], 1e-6)
        pace.update(sim=now_sim, wall=now_wall)
        return f"event: {tag} (sim {now_sim:.1f} s, {rate:.2f} x real time since the last event)"

    def on_event(tag: str) -> None:
        if tag.startswith("asked_"):
            return
        if tag == "grip_keeper":  # a force grip let go: how the keeper held it through the carry
            entry = holder["ctx"].grip_keepers[-1]
            status.log(keeper_line(entry["part"], entry) or f"grip keeper {entry['part']}: no summary")
            return
        status.log(paced(tag))
        if tag.startswith("bump_"):  # every bump the arm felt, and how much higher it now crosses because of them
            status.set_sensor("Arm bumps", bump_rows(holder["ctx"].bumps))

    ctx = SortContext(layout=layout, episode=episode, ladder=ladder, logbook=logbook, fault=None if args.fault == "none" else args.fault, on_event=on_event, **grip_settings(args.grip), close_fraction=args.close, dance=args.dance,
                      scales=TrayScales(layout, seed=args.seed) if args.scales else None)
    holder["ctx"] = ctx
    if film is not None:
        film.start()
        status.log(f"filming {', '.join(film.names)} at {film.fps:g} fps of sim time into {film.out}")
    print(f"parts this episode: " + ", ".join(f"{p} ({episode.truth(p).label})" for p in episode.parts))
    try:
        result = run_sort(sim, ctx)
        print(result.trace.text())
        if result.status.value != "success":  # a recovered failure is already in the log as its events and answers
            for line in result.trace.text().splitlines():
                if "failure" in line:
                    status.log("failed step: " + line.strip())
        line = finish_line(result.status.value, result.checks, result.decisions)
        status.log(line)
        print(f"\n{line}")
        if film is not None:  # a film run ends with the sort: holding the viewer would keep saving frames
            film.stop()
            status.log(f"film: {film.frames} frames an angle in {film.out}")
            return 0
        print("holding the viewer open; Ctrl+C to quit")
        while True:
            sim.step(10)
    except KeyboardInterrupt:
        status.log("run stopped (Ctrl+C)")
    finally:
        if film is not None:
            film.stop()
        sort_module.ask = real_ask
        SKILLS["grasp"] = real_grasp
        logbook.close()
        status.close()
        page.close()
        if panel is not None:
            panel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
