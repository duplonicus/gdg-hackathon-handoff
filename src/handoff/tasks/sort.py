"""Stage 2 task: sort the parts, asking the ladder at the marked points.

The tree, in words:

    drive to the table, look down
    for each part:
        feed it into the pick station (the stop-and-go line)
        INSPECT: ask "finish"  -> good bin or rework bin
        ASK "safe_to_grasp"; if grasp: pick (reach, approach, grasp, lift), else hold or skip
        carry to the chosen bin and release (a forced drop can be injected here)
        ASK "in_bin" (the tray scales first, when fitted; the camera model and a person only when the scales
        cannot tell); if no: recover (a part on the table is re-fed and placed again, once; after that, or for a
        part in the other tray, the operator places it), up to twice
    ASK "sort_complete"; only if yes (and SortContext.dance is on): back away from the table and dance

Every question has a fixed option list, the ladder decides, the answer is checked for
executability before anything moves, safety-type questions also get a hard rule from sim state,
and every decision is logged with the truth the sim knows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..bt import Action, Condition, ForEach, Retry, Selector, Sequence, Status, Trace
from ..checks import WorldState, any_on_floor, in_bin, on_floor, snapshot
from ..decider import QUESTIONS, Decision, Ladder, Question
from ..logbook import Logbook, encode_png_b64
from ..perception import locate_bins
from ..scene import Episode, Layout, episode_layout
from ..sensors import TrayScales, scale_verdict
from ..sim.base import Arm, Simulator
from ..skills import SKILLS, TaskContext, bin_seen, grasp_wrist_pos, let_go, look_at, open_hand, wait_until_resting

# Robot re-picks of one part before the operator places it instead. The Oct 21 demo (seed 7, closure 0.75) dropped
# part_12 on the carry, re-picked it from the table and dropped it again, four times: a re-pick that failed once tends
# to fail the same way, and the operator is the hand-off for what the robot cannot do.
MAX_ROBOT_REPICKS = 1

QUESTION_CAMERA = {"finish": "inspect", "safe_to_grasp": "debug", "in_bin": "bins", "sort_complete": "bins"}


@dataclass
class SortContext:
    layout: Layout
    episode: Episode
    ladder: Ladder
    logbook: Logbook | None = None
    arm: Arm = "left"
    grasp_mode: str = "fingers"
    grasp_control: str = "fixed"  # how the hand closes: "fixed" or "force" (skills.TaskContext.grasp_control)
    grip: str | None = None  # the --grip choice by name, for the run log (skills.TaskContext.grip)
    close_fraction: float | None = None  # the fixed grip's closure; None: TaskContext's default
    grip_thumb_n: float | None = None  # force grip: close the thumb on its own to this (N); None: it follows the fingers
    fault: str | None = None  # "drop_on_carry": open the hand mid-carry on the first part; "crowded": a second part lands 5 cm from the first;
    # "wrong_bin": the first part is set down in the other tray (the landing the camera model calls "in the bin")
    on_event: Callable[[str], None] | None = None
    decisions: list[Decision] = field(default_factory=list)
    intended_bin: dict[str, str] = field(default_factory=dict)
    fault_fired: bool = False
    hold_steps: int = 0
    bins_seen: dict[str, tuple[float, float]] = field(default_factory=dict)  # where the robot saw each bin this episode
    nominal_layout: Layout | None = None  # the drawing; `layout` becomes the truth for this episode
    return_home: bool = True  # False: after a release the arm only lifts clear and starts back (the retract skill)
    scales: TrayScales | None = None  # a load cell under each tray; None: the camera alone answers in_bin
    scale_tare: dict[str, float] | None = None  # tray readings (g) taken before the current release
    scale_now: dict[str, Any] | None = None  # the scales' verdict for the in_bin question being asked
    operator_moves: int = 0  # parts the operator set in their right tray (from the other tray, or after a failed re-pick)
    robot_repicks: dict[str, int] = field(default_factory=dict)  # part -> times the robot re-picked it from the table
    # The macarena after a verified sort. Off by default (dup, Oct 21: "no dancing anymore until i'm happy with
    # everything"); when on, the robot backs away from the table first (skills.back_away_from_table).
    dance: bool = False
    # Every time the hand touched the table or a tray during a wrist move (skills.back_off). One list for the whole run,
    # handed to every part's TaskContext, so each bump raises the crossing height for the parts after it.
    bumps: list[dict[str, Any]] = field(default_factory=list)
    # One entry per force grip, at its release: the part and how the grip keeper held it (skills.GripKeeper.summary).
    grip_keepers: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ground truth (the in-bin checks, the labels) uses where the bins truly are this episode. The robot does not:
        # it aims at bins_seen, filled by the locate-bins step.
        self.nominal_layout = self.layout
        self.layout = episode_layout(self.layout, self.episode)

    def task_ctx(self, part: str, target_bin: str = "good") -> TaskContext:
        already_there = sum(1 for p, b in self.intended_bin.items() if b == target_bin and p != part)
        return TaskContext(
            layout=self.nominal_layout, target_part=part, parts=self.episode.parts, target_bin=target_bin, arm=self.arm,
            grasp_mode=self.grasp_mode, grasp_control=self.grasp_control, grip=self.grip, on_event=self.on_event, place_slot=already_there,
            bins_seen=self.bins_seen, bumps=self.bumps, grip_thumb_n=self.grip_thumb_n,
            **({"close_fraction": self.close_fraction} if self.close_fraction is not None else {}),
        )


# ----- what the robot can say about the world without cheating -----


def state_text(sim: Simulator, ctx: SortContext, qid: str, part: str) -> str:
    layout = ctx.layout
    wrist = sim.wrist_pose(ctx.arm).pos
    closure = sim.hand_closure(ctx.arm)
    hand = "closed" if closure > 0.5 else "open"
    slot = layout.slot_pos(0)
    bins = (
        "In the camera over the bins, the orange tray on the left is the rework bin and the green tray on the right is the good bin. "
        "Parts are painted blue; bare metal is grey."
    )
    if qid == "finish":
        return (
            f"Robot parked at the parts table. A part stands in the pick station at x {slot[0]:.2f} y {slot[1]:.2f} under the inspection camera. "
            "Parts are painted blue. A good part is one uniform blue on every face. Lines, patches or areas of any other colour make it defective."
        )
    if qid == "safe_to_grasp":
        target = grasp_wrist_pos(sim, ctx.task_ctx(part))
        d = float(np.linalg.norm(target - wrist))
        others = [p for p in ctx.episode.parts if p != part]
        near = min((float(np.linalg.norm(sim.object_state(p).pos[:2] - sim.object_state(part).pos[:2])) for p in others), default=9.9)
        return (
            f"Left hand {hand}, wrist {d * 100:.0f} cm from the planned grasp pose beside the part in the pick station. "
            f"Nearest other part is {near * 100:.0f} cm away. The hand must close on the part from the side with the part against the palm."
        )
    if qid == "in_bin":
        bin_name = ctx.intended_bin.get(part, "good")
        colour = "green (right)" if bin_name == "good" else "orange (left)"
        return f"The hand released the part over the {bin_name} bin, the {colour} tray, about {ctx.hold_steps * sim.dt:.1f} s ago and is now {hand}. {bins}"
    if qid == "sort_complete":
        placed = ", ".join(f"{p} to {b}" for p, b in ctx.intended_bin.items()) or "nothing yet"
        return f"The robot has tried to place: {placed}. {len(ctx.episode.parts)} parts were fed. {bins}"
    return ""


def frame_b64(sim: Simulator, qid: str) -> tuple[np.ndarray | None, str | None]:
    frame = sim.capture(QUESTION_CAMERA[qid])
    if frame.rgb is None:
        return None, None
    return frame.rgb, encode_png_b64(frame.rgb)


# ----- checks from sim state: executability, hard rules, and the truth for the log -----


def world(sim: Simulator, ctx: SortContext) -> WorldState:
    return snapshot(sim, ctx.layout, (ctx.arm,), ctx.episode.parts)


def part_on_table(sim: Simulator, ctx: SortContext, part: str) -> bool:
    """Standing or lying on the table top, outside every bin: the only place the line can re-feed from."""
    state = world(sim, ctx)
    pos = state.objects[part].pos
    (x0, x1), (y0, y1) = ctx.layout.table_x, ctx.layout.table_y
    on_top = x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1 and ctx.layout.table_top - 0.01 <= pos[2] - 0.02 <= ctx.layout.table_top + 0.06
    return bool(on_top) and not any(in_bin(state, part, b) for b in ctx.layout.bin_names)


def part_in_station(sim: Simulator, ctx: SortContext, part: str, tol: float = 0.06) -> bool:
    pos = sim.object_state(part).pos
    slot = ctx.layout.slot_pos(0)
    return float(np.hypot(pos[0] - slot[0], pos[1] - slot[1])) <= tol and abs(float(pos[2]) - slot[2]) <= 0.03


def executable(sim: Simulator, ctx: SortContext, part: str) -> Callable[[Question], dict[str, bool]]:
    def check(question: Question) -> dict[str, bool]:
        can = {o: True for o in question.options}
        if question.id == "safe_to_grasp":
            hand_open = sim.hand_closure(ctx.arm) < 0.2
            can["grasp"] = hand_open and part_in_station(sim, ctx, part)
        return can

    return check


def rules(sim: Simulator, ctx: SortContext, part: str) -> Callable[[Question], str | None]:
    def rule(question: Question) -> str | None:
        if question.kind != "safety":
            return None
        state = world(sim, ctx)
        if question.id == "safe_to_grasp":
            if on_floor(state, part):
                return "abort"
            others = [p for p in ctx.episode.parts if p != part]
            too_close = any(float(np.linalg.norm(state.objects[p].pos[:2] - state.objects[part].pos[:2])) < 0.06 for p in others)
            if too_close:
                return "abort"
        if question.id == "sort_complete" and any_on_floor(state):
            return "no"
        return None

    return rule


def truth(sim: Simulator, ctx: SortContext, part: str, qid: str) -> str | None:
    state = world(sim, ctx)
    if qid == "finish":
        return "good" if ctx.episode.truth(part).kind == "good" else "defective"  # the kind of defect stays in the log context
    if qid == "safe_to_grasp":
        return "grasp" if executable(sim, ctx, part)(QUESTIONS[qid])["grasp"] and rules(sim, ctx, part)(QUESTIONS[qid]) is None else "abort"
    if qid == "in_bin":
        intended = ctx.intended_bin.get(part, "good")
        return "yes" if intended in ctx.layout.bin_names and in_bin(state, part, intended) else "no"
    if qid == "sort_complete":
        return "yes" if all(any(in_bin(state, p, b) for b in ctx.layout.bin_names) for p in ctx.episode.parts) else "no"
    return None


def read_scales(sim: Simulator, ctx: SortContext, part: str) -> dict[str, Any] | None:
    """The tray scales' answer for the part just released: read now, compare with the tare taken before the release."""
    if ctx.scales is None or ctx.scale_tare is None:
        return None
    reading, true = ctx.scales.read(sim)
    intended = ctx.intended_bin.get(part, "good")
    verdict, where = scale_verdict(ctx.scale_tare, reading, intended, ctx.scales.part_grams)
    return {
        "verdict": verdict, "where": where, "intended": intended, "part_g": round(ctx.scales.part_grams, 1),
        "tare_g": ctx.scale_tare, "reading_g": reading, "true_g": {b: round(g, 2) for b, g in true.items()},
    }


def ask(sim: Simulator, ctx: SortContext, qid: str, part: str) -> Decision:
    question = QUESTIONS[qid]
    # read the scales before the frame, so the frame and the readings show the same moment (a reading takes 0.2 s)
    ctx.scale_now = read_scales(sim, ctx, part) if qid == "in_bin" else None
    rgb, b64 = frame_b64(sim, qid)
    frame_path = ctx.logbook.save_frame(rgb, qid) if (ctx.logbook and rgb is not None) else None
    ctx.ladder.rules = rules(sim, ctx, part)
    context: dict = {"part": part, "finish_truth": ctx.episode.truth(part).label, "intended_bin": ctx.intended_bin.get(part)}
    if qid in ("in_bin", "sort_complete"):
        state = world(sim, ctx)
        context["parts"] = {
            p: {"pos": [round(float(v), 3) for v in state.objects[p].pos], "speed": round(float(np.linalg.norm(state.objects[p].lin_vel)), 3),
                "in": [b for b in ctx.layout.bin_names if in_bin(state, p, b)]}
            for p in ctx.episode.parts
        }
        # where the bins truly were and where the robot saw them, so the labels can be recomputed offline
        context["bins_true"] = {name: [round(cx, 4), round(cy, 4)] for name, cx, cy in ctx.layout.bin_xy}
        context["bins_seen"] = {name: [round(x, 4), round(y, 4)] for name, (x, y) in ctx.bins_seen.items()}
    if ctx.scale_now is not None:
        context["scale"] = ctx.scale_now
    decision = ctx.ladder.decide(
        question, state_text(sim, ctx, qid, part), b64, executable(sim, ctx, part), t_sim=sim.time, frame_path=frame_path, context=context,
    )
    decision.truth = truth(sim, ctx, part, qid)
    ctx.decisions.append(decision)
    if ctx.logbook:
        ctx.logbook.record(decision)
    if ctx.on_event:
        ctx.on_event(f"asked_{qid}_{decision.answer}")
    return decision


# ----- the line feed and the dance -----


REJECT_LANE = (0.75, -0.25)  # where the line puts a part the robot would not pick (x, y on the table)


def reject_part(sim: Simulator, ctx: SortContext, part: str) -> None:
    """The line moves a held-back part out of the station to the reject lane, clearing it."""
    z = ctx.layout.part_center_z
    sim.set_object_pose(part, (REJECT_LANE[0], REJECT_LANE[1], z), 0.0)
    sim.settle(0.3)


def feed_part(sim: Simulator, ctx: SortContext, part: str, seconds: float = 1.0) -> None:
    """Slide the next part into the pick station along the row: the stop-and-go line."""
    start = sim.object_state(part).pos.copy()
    end = np.array(ctx.layout.slot_pos(0))  # standing upright at the station height, however it was lying
    n = max(1, int(seconds / sim.dt / 5))
    for k in range(1, n + 1):
        p = start + (end - start) * k / n
        sim.set_object_pose(part, (float(p[0]), float(p[1]), float(p[2])), 0.0)  # the line squares the part up
        sim.step(5)
    sim.settle(0.3)


# ----- the tree -----


def build_tree(sim: Simulator, ctx: SortContext) -> Sequence:
    def skill(name: str, bin_key: str | None = None) -> Callable[[dict], bool]:
        def run(bb: dict) -> bool:
            tctx = ctx.task_ctx(bb.get("part", ctx.episode.parts[0]), bb.get("bin", "good"))
            result = SKILLS[name](sim, tctx)
            bb["note"] = result.reason
            return result.ok

        return run

    def watch_station() -> None:
        look_at(sim, ctx.layout.slot_pos(0))

    def watch_bin(name: str) -> None:
        region = bin_seen(ctx.task_ctx(ctx.episode.parts[0]), name)
        look_at(sim, (region.cx, region.cy, region.floor_z))

    def find_bins(bb: dict) -> bool:
        """Find both trays in the bins camera before anything is carried: the bins are not where the drawing says."""
        rgb, xyz = sim.capture_cloud("bins")
        seen = locate_bins(rgb, xyz, ctx.nominal_layout)
        missing = [b for b in ctx.nominal_layout.bin_names if b not in seen]
        if missing:
            bb["note"] = f"could not find the {' and '.join(missing)} bin in the bins camera"
            return False
        ctx.bins_seen.update(seen)
        sim.bins_seen = ctx.bins_seen  # the bump guard names the tray it felt from where the robot saw them, not the drawing
        return True

    def feed(bb: dict) -> bool:
        watch_station()
        if not part_in_station(sim, ctx, bb["part"]):
            feed_part(sim, ctx, bb["part"])
        if ctx.fault == "crowded" and not ctx.fault_fired and bb["part"] == ctx.episode.parts[0] and len(ctx.episode.parts) > 1:
            # the forced failure: the next part arrives early and stops 5 cm from the one in the station
            ctx.fault_fired = True
            x, y, z = sim.object_state(bb["part"]).pos
            sim.set_object_pose(ctx.episode.parts[1], (float(x) + 0.05, float(y), float(z)), 0.0)
            sim.settle(0.3)
            if ctx.on_event:
                ctx.on_event("fault_crowded")
        return part_in_station(sim, ctx, bb["part"])

    def hold(bb: dict) -> bool:
        """Not safe to grasp: leave the part to the line's reject lane and carry on; the sort will not complete."""
        reject_part(sim, ctx, bb["part"])
        ctx.intended_bin[bb["part"]] = "reject"
        return False

    def inspect(bb: dict) -> bool:
        d = ask(sim, ctx, "finish", bb["part"])
        bb["bin"] = "good" if d.answer == "good" else "rework"
        ctx.intended_bin[bb["part"]] = bb["bin"]
        return True

    def safe(bb: dict) -> bool:
        tctx = ctx.task_ctx(bb["part"])
        SKILLS["reach"](sim, tctx)  # bring the open hand to the pre-grasp pose first, so the frame shows it
        d = ask(sim, ctx, "safe_to_grasp", bb["part"])
        bb["safe"] = d.answer
        return d.answer == "grasp"

    def unweld(part: str) -> None:
        """A pick or re-pick that failed after the grasp welded the part must let it go, or the part rides the wrist
        into the reject lane, a tray or the next part's pick (found by the Oct 21 review)."""
        if ctx.grasp_mode == "weld":
            sim.detach(part)

    def pick(bb: dict) -> bool:
        tctx = ctx.task_ctx(bb["part"], bb["bin"])
        for name in ("approach", "grasp", "lift"):
            r = SKILLS[name](sim, tctx)
            if ctx.on_event:
                ctx.on_event(f"pick_{name}")
            if not r.ok:
                bb["note"] = f"{name}: {r.reason}"
                unweld(bb["part"])
                return False
        return True

    def tare() -> None:
        ctx.scale_tare = ctx.scales.read(sim)[0] if ctx.scales is not None else None

    def carry_and_release(bb: dict) -> bool:
        tctx = ctx.task_ctx(bb["part"], bb["bin"])
        tare()  # the part is in the hand, so neither tray carries it yet
        watch_bin(bb["bin"])
        if ctx.fault == "drop_on_carry" and not ctx.fault_fired and bb["part"] == ctx.episode.parts[0]:
            ctx.fault_fired = True
            here = sim.wrist_pose(ctx.arm).pos
            sim.move_wrist(ctx.arm, here + np.array([0.0, 0.08, 0.0]), sim.wrist_pose(ctx.arm).quat, timeout_s=1.0, tol_m=0.02)
            if ctx.grasp_mode == "weld":
                sim.detach(bb["part"])
            let_go(sim, ctx.arm)  # the forced failure: the part drops short of the bin
            sim.settle(1.0)
            if ctx.on_event:
                ctx.on_event("fault_drop")
        else:
            place = tctx
            if ctx.fault == "wrong_bin" and not ctx.fault_fired and bb["part"] == ctx.episode.parts[0]:
                ctx.fault_fired = True
                other = next(b for b in ctx.layout.bin_names if b != bb["bin"])
                place = ctx.task_ctx(bb["part"], other)  # the forced failure: set down over the other tray
                if ctx.on_event:
                    ctx.on_event("fault_wrong_bin")
            r = SKILLS["move_to_bin"](sim, place)
            if r.ok:
                r = SKILLS["lower"](sim, place)  # down into the bin as far as the hand allows, so the part is set down, not dropped
            if not r.ok:
                bb["note"] = r.reason
            if ctx.grasp_mode == "weld":
                sim.detach(bb["part"])
            keeper = open_hand(sim, tctx)
            if keeper is not None:
                ctx.grip_keepers.append({"part": bb["part"], **keeper.summary()})
                if ctx.on_event:
                    ctx.on_event("grip_keeper")
        ctx.hold_steps = int(1.5 / sim.dt)
        wait_until_resting(sim, tctx, max_s=3.0, min_s=1.5)
        SKILLS["go_home" if ctx.return_home else "retract"](sim, tctx)
        watch_bin(bb["bin"])  # still on the bin for the in_bin question
        return True

    def landed(bb: dict) -> bool:
        d = ask(sim, ctx, "in_bin", bb["part"])
        return d.answer == "yes"

    def operator_rebin(bb: dict, event: str = "operator_rebin") -> bool:
        """The part landed in the other tray, or the robot's re-pick already failed: the operator picks it up and sets
        it in the right tray (``event``: operator_rebin from the other tray, operator_place after a failed re-pick).

        A deterministic hand-off, not a model call: the scales (or the camera's answer) said the part is not where it
        belongs. The robot does not reach into a tray
        (its grasp is a side grasp of a part standing in the station); a robot re-pick is a later step."""
        part, target = bb["part"], bb["bin"]
        # lifted out (in the sim: set aside off the table), then the scales tare, so the check after sees one landing
        aside = (ctx.layout.park_xy[0], ctx.layout.park_xy[1] - 0.5)  # a floor spot clear of the parked pool row
        sim.set_object_pose(part, (*aside, ctx.layout.part_size[2] / 2.0), 0.0)
        sim.settle(0.3)
        tare()
        region = ctx.layout.bin_region(target)
        state = world(sim, ctx)
        others = [state.objects[p].pos[:2] for p in ctx.episode.parts if p != part and in_bin(state, p, target)]
        spots = [(region.cx + dx, region.cy + dy) for dx in (-0.04, 0.04) for dy in (-0.04, 0.04)]
        spot = max(spots, key=lambda xy: min((float(np.hypot(xy[0] - o[0], xy[1] - o[1])) for o in others), default=9.9))
        sim.set_object_pose(part, (spot[0], spot[1], region.floor_z + ctx.layout.part_size[2] / 2.0 + 0.005), 0.0)
        sim.settle(1.0)
        ctx.operator_moves += 1
        if ctx.on_event:
            ctx.on_event(event)
        watch_bin(target)
        return True

    def recover(bb: dict) -> bool:
        """The part is not in its bin: find it (where the sim says it is) and bring it to the station."""
        part = bb["part"]
        if ctx.scale_now is not None and ctx.scale_now.get("where") in ("other bin", "no bin"):
            wrong_tray = ctx.scale_now["where"] == "other bin"  # the scales weighed it in the other tray
        else:  # no scales, or they could not tell: where the sim says it is, as for a part on the table
            state = world(sim, ctx)
            wrong_tray = any(in_bin(state, part, b) for b in ctx.layout.bin_names if b != bb["bin"])
        if wrong_tray:
            return operator_rebin(bb)
        if not part_on_table(sim, ctx, part):
            bb["note"] = "part is not on the table (floor, or already in a bin): nothing the line can re-feed"
            return False
        if ctx.robot_repicks.get(part, 0) >= MAX_ROBOT_REPICKS:
            return operator_rebin(bb, "operator_place")
        ctx.robot_repicks[part] = ctx.robot_repicks.get(part, 0) + 1
        watch_station()
        feed_part(sim, ctx, part)
        tctx = ctx.task_ctx(part, bb["bin"])
        SKILLS["reach"](sim, tctx)
        for name in ("approach", "grasp", "lift"):
            r = SKILLS[name](sim, tctx)
            if ctx.on_event:
                ctx.on_event(f"recover_{name}")
            if not r.ok:
                bb["note"] = f"recover {name}: {r.reason}"
                unweld(part)
                return False
        return carry_and_release(bb)

    def complete(bb: dict) -> bool:
        d = ask(sim, ctx, "sort_complete", ctx.episode.parts[-1])
        return d.answer == "yes"

    def recover_strays(bb: dict) -> bool:
        """The sort is not complete: any part that is not in a bin and not rejected goes round again."""
        state = world(sim, ctx)
        strays = [p for p in ctx.episode.parts if not any(in_bin(state, p, b) for b in ctx.layout.bin_names) and ctx.intended_bin.get(p) != "reject"]
        if not strays:
            return False
        for part in strays:
            bb["part"], bb["bin"] = part, ctx.intended_bin.get(part, "good")
            if not recover(bb):
                return False
        return True

    def dance(bb: dict) -> bool:
        return SKILLS["macarena"](sim, ctx.task_ctx(ctx.episode.parts[-1])).ok

    per_part = Sequence(
        "sort one part",
        Action("feed into station", feed),
        Action("inspect: ask finish", inspect),
        Selector(
            "pick or hold",
            Sequence("ask safe_to_grasp, then pick", Retry("safe?", Condition("safe_to_grasp == grasp", safe), 2), Action("pick", pick)),
            Action("hold: send the part to the reject lane", hold),
        ),
        Action("carry and release", carry_and_release),
        Retry(
            "placed?",
            Selector("verify or recover", Condition("ask in_bin == yes", landed), Sequence("recover", Action("re-pick and place", recover), Condition("ask in_bin == yes", landed))),
            2,
        ),
    )
    return Sequence(
        "sort the parts",
        Action("drive_to_table", skill("drive_to_table")),
        Action("look_at_table", skill("look_at_table")),
        Action("find the bins", find_bins),
        ForEach("parts", "parts", "part", per_part),
        Retry("complete?", Selector("verify or recover strays", Condition("ask sort_complete == yes", complete), Sequence("recover strays", Action("re-place strays", recover_strays), Condition("ask sort_complete == yes", complete))), 2),
        *([Action("macarena", dance)] if ctx.dance else []),
    )


@dataclass
class SortResult:
    status: Status
    trace: Trace
    decisions: list[Decision]
    checks: dict[str, Any]
    sim_time: float


def run_sort(sim: Simulator, ctx: SortContext) -> SortResult:
    bb: dict[str, Any] = {"parts": list(ctx.episode.parts)}
    trace = Trace()
    status = build_tree(sim, ctx).tick(bb, trace)
    state = world(sim, ctx)
    right = {p: "rework" if ctx.episode.truth(p).defective else "good" for p in ctx.episode.parts}
    checks = {
        "parts_in_correct_bin": sum(1 for p in ctx.episode.parts if in_bin(state, p, right[p])),
        "parts_in_any_bin": sum(1 for p in ctx.episode.parts if any(in_bin(state, p, b) for b in ctx.layout.bin_names)),
        "n_parts": len(ctx.episode.parts),
        "on_floor": any_on_floor(state),
        "danced": any(n == "macarena" and s == "success" for _, n, s in trace.lines),
        "operator_moves": ctx.operator_moves,
        "mis_sorted": {p: where_is(state, ctx, p, right[p]) for p in ctx.episode.parts if not in_bin(state, p, right[p])},
        "bumps": len(getattr(ctx, "bumps", [])),  # the hand touching the table or a tray box during a wrist move (skills)
    }
    return SortResult(status, trace, list(ctx.decisions), checks, sim.time)


def where_is(state: WorldState, ctx: SortContext, part: str, right: str) -> str:
    """Where a part that is not in its right bin ended up, in words for the run's last line."""
    wrong = [b for b in ctx.layout.bin_names if in_bin(state, part, b)]
    if wrong:
        return f"in the {wrong[0]} bin, belongs in {right}"
    if ctx.intended_bin.get(part) == "reject":
        return "held back to the reject lane"
    if on_floor(state, part):
        return "on the floor"
    return "not in a bin"
