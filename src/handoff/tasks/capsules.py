"""One shift at the capsule cells, as a behaviour tree (Nov 4).

1. **Where am I needed?** Both cells' overview cameras, side by side: cell a, cell b, or neither.
2. Drive there and tuck in under the table at the station: the only drive until the shift backs out (Nov 5).
3. **Scan the tray:** perception (no model) finds every capsule; they are grouped into blocks of ``block`` capsules
   and each block is asked **leak_block** ("is any capsule here a leaker?"). Each capsule of a block that says yes is
   asked **leak_capsule** on a crop centred on it.
4. For each capsule called a leaker: the hand on its side weld-picks it and drops it in that side's leaker bin, and
   **leaker_landed** is asked on that bin camera's before and after pictures.
5. **tray_clear** on the whole tray: if no, scan once more; then back out.

Every question goes up the same ladder as the sorting task (rules, models, a person, the safe option) and is logged
with its picture, every rung's answer and the simulator's truth, so any model can be replayed on it afterwards. The
tray picture and every capsule's pixel box are logged too, so crops of any block size can be made from the log
without running the robot again (the "how many capsules per question" curve).

The tray dump into the good bin is not in this tree yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..bt import Action, Condition, Selector, Sequence, Status, Trace
from ..capsule_cell import TwoCells
from ..capsule_episode import CapsuleEpisode
from ..capsule_perception import SeenCapsule, find_capsules
from .. import capsule_skills
from ..capsule_cell import BIN_SIDES, parked_pos
from ..capsule_skills import _to_local, base_drift, drop_in_bin, park_station, pick_capsule, side_of, travel_pose
from ..decider import QUESTIONS, Decision, Ladder
from ..drive import back_out
from ..logbook import Logbook, encode_png_b64
from ..sim.base import Simulator, yaw_of

BLOCK = 16  # capsules per leak_block question in a live run: 4 along the tray's depth by 4 across
CAPSULE_CROP_M = 0.045  # the leak_capsule picture spans this much paper around its capsule (its pool and its neighbours)
PICK_MATCH_M = 0.012  # the capsule the hand closes on is the one within this of where the camera saw it


@dataclass
class CapsuleContext:
    cells: TwoCells
    episode: CapsuleEpisode
    ladder: Ladder
    logbook: Logbook | None = None
    block: int = BLOCK
    decisions: list[Decision] = field(default_factory=list)
    on_event: Callable[[str], None] | None = None
    see: Callable[[Simulator, "CapsuleContext"], tuple[np.ndarray | None, list[SeenCapsule]]] | None = None
    truth_now: str | None = None  # the truth of the question being asked, for an oracle rung
    cell: str | None = None  # where the robot went
    removed: list[str] = field(default_factory=list)  # every object the robot put in a leaker bin
    move_failures: list[dict] = field(default_factory=list)
    base_home: tuple[float, float, float] | None = None  # the base pose when it parked (x, y, yaw)
    base_drift_mm: float = 0.0  # the most the base moved from there during the shift (a sensor: shown in the run log)
    base_drift_deg: float = 0.0  # and the most it turned
    loosened: list[str] = field(default_factory=list)  # every loose body swapped in for a fixed capsule this shift  # every pick or drop that failed, with its reason (shown in the run log)
    spares_used: int = 0
    parked: bool = False


# ----- what is true, from the simulator (for the log and the oracle; decisions never read it) -----


def on_paper(ctx: CapsuleContext, pos) -> bool:
    """Lying on the parchment of the cell with work, at a capsule's resting height."""
    lay = ctx.cells.layout
    cell = ctx.cells.cell(ctx.episode.work_cell or "a")
    u, v = _to_local(cell, float(pos[0]), float(pos[1]))
    flat = abs(float(pos[2]) - (lay.paper_top + lay.capsule[2] / 2.0)) <= 0.01
    return flat and abs(u - lay.tray_center[0]) <= lay.paper_half[0] and abs(v - lay.tray_center[1]) <= lay.paper_half[1]


def leakers_on_tray(sim: Simulator, ctx: CapsuleContext) -> list[str]:
    """The leakers still lying on the work tray: the fixed capsule at a leaker spot until the pick swaps it out, and
    the loose leaker body if one ever lands back on the paper."""
    if ctx.episode.work_cell is None:
        return []
    out = []
    for k, spot in enumerate(ctx.episode.leakers):
        for name in (ctx.cells.grid_names[spot], ctx.cells.leaker_names[k]):
            if on_paper(ctx, sim.object_state(name).pos):
                out.append(name)
    return out


def leaker_near(sim: Simulator, ctx: CapsuleContext, xy, radius: float = PICK_MATCH_M) -> str | None:
    for name in leakers_on_tray(sim, ctx):
        pos = sim.object_state(name).pos
        if float(np.hypot(pos[0] - xy[0], pos[1] - xy[1])) <= radius:
            return name
    return None


def leaker_owners(sim: Simulator, ctx: CapsuleContext, caps: list[SeenCapsule]) -> dict[int, str]:
    """Which seen capsule each leaker on the tray is: the nearest one, within PICK_MATCH_M. One capsule per leaker: a
    loose leaker can settle a few millimetres toward a neighbour, and a radius alone labelled both (Nov 4 smoke run)."""
    owners: dict[int, str] = {}
    for name in leakers_on_tray(sim, ctx):
        pos = sim.object_state(name).pos
        dists = [float(np.hypot(c.xyz[0] - pos[0], c.xyz[1] - pos[1])) for c in caps]
        if dists and min(dists) <= PICK_MATCH_M:
            owners[int(np.argmin(dists))] = name
    return owners


def leaker_slot(ctx: CapsuleContext, name: str | None) -> int | None:
    """Which of the episode's leakers an object is: its fixed capsule at the spot, or its loose body."""
    if name in ctx.cells.leaker_names:
        return ctx.cells.leaker_names.index(name)
    if name in ctx.cells.grid_names and ctx.cells.grid_names.index(name) in ctx.episode.leakers:
        return ctx.episode.leakers.index(ctx.cells.grid_names.index(name))
    return None


def in_leaker_bin(sim: Simulator, ctx: CapsuleContext, name: str) -> bool:
    lay = ctx.cells.layout
    cell = ctx.cells.cell(ctx.cell or "a")
    pos = sim.object_state(name).pos
    u, v = _to_local(cell, float(pos[0]), float(pos[1]))
    floor = lay.table_top + lay.leaker_bin_floor
    if not floor <= float(pos[2]) <= floor + lay.leaker_bin_wall:
        return False
    return any(abs(u - bx) <= lay.leaker_bin_inner[0] / 2 and abs(v - by) <= lay.leaker_bin_inner[1] / 2 for bx, by in (lay.leaker_bin(s) for s in BIN_SIDES))


# ----- pictures -----


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    """Two pictures in one, left and right, with a white gap: models take one image, and the question says which is which."""
    h = max(left.shape[0], right.shape[0])
    pad = lambda img: np.pad(img, ((0, h - img.shape[0]), (0, 0), (0, 0)), constant_values=255)
    return np.concatenate([pad(left), np.full((h, gap, 3), 255, dtype=np.uint8), pad(right)], axis=1)


def padded(rgb: np.ndarray, box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    """The box grown by ``pad`` pixels each way and kept inside the picture."""
    top, left, bottom, right = box
    return (max(0, top - pad), max(0, left - pad), min(rgb.shape[0], bottom + pad), min(rgb.shape[1], right + pad))


def crop(rgb: np.ndarray, box: tuple[int, int, int, int], pad: int) -> np.ndarray:
    top, left, bottom, right = padded(rgb, box, pad)
    if bottom <= top or right <= left:
        return rgb[:1, :1]
    return rgb[top:bottom, left:right]


def in_view(caps: list[SeenCapsule], window: tuple[int, int, int, int]) -> list[SeenCapsule]:
    """Every capsule whose pixels reach into the window: what a picture cut to that window shows, even in part."""
    top, left, bottom, right = window
    return [c for c in caps if c.box[0] < bottom and c.box[2] > top and c.box[1] < right and c.box[3] > left]


def wholly_in(cap: SeenCapsule, window: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = window
    return cap.box[0] >= top and cap.box[1] >= left and cap.box[2] <= bottom and cap.box[3] <= right


def block_truth(caps: list[SeenCapsule], window: tuple[int, int, int, int], leaky: set[int]) -> str | None:
    """yes: a leaker lies wholly in the picture; no: no leaker shows at all; None: a leaker shows only in part, at the
    edge. A sliver of a capsule with half a pool is neither a clear yes nor a clear no, and counting it as yes made
    small blocks score worse than big ones (first replays, Nov 5): such pictures are left out of scoring."""
    shown = [caps.index(c) for c in in_view(caps, window)]
    if any(i in leaky and wholly_in(caps[i], window) for i in shown):
        return "yes"
    return None if any(i in leaky for i in shown) else "no"


def block_box(caps: list[SeenCapsule]) -> tuple[int, int, int, int]:
    return (min(c.box[0] for c in caps), min(c.box[1] for c in caps), max(c.box[2] for c in caps), max(c.box[3] for c in caps))


def capsule_box(cap: SeenCapsule, rgb: np.ndarray, px_per_m: float) -> tuple[int, int, int, int]:
    """A square box centred on the capsule's pixels, ``CAPSULE_CROP_M`` of paper across."""
    r = (cap.box[0] + cap.box[2]) // 2
    c = (cap.box[1] + cap.box[3]) // 2
    half = int(round(CAPSULE_CROP_M * px_per_m / 2))
    return (r - half, c - half, r + half, c + half)


def pixels_per_metre(caps: list[SeenCapsule], length_m: float) -> float:
    """From the capsules themselves: their median pixel extent over their known length (the longer box side)."""
    if not caps:
        return 1.0
    sizes = [max(c.box[2] - c.box[0], c.box[3] - c.box[1]) for c in caps]
    return float(np.median(sizes)) / length_m


def blocks(caps: list[SeenCapsule], ctx: CapsuleContext) -> list[list[SeenCapsule]]:
    """Group the capsules into blocks of about ``ctx.block`` neighbours: square patches of the tray's grid, in the
    tray's own frame (so cell b, turned round, groups the same way)."""
    if ctx.block <= 1:
        return [[c] for c in caps]
    lay = ctx.cells.layout
    cell = ctx.cells.cell(ctx.cell or "a")
    side = int(round(np.sqrt(ctx.block)))
    keyed: dict[tuple[int, int], list[SeenCapsule]] = {}
    # Each capsule takes the row and column of the nearest grid spot, counted along the grid's own rows and columns: the
    # aisle between the half-trays is not a run of empty columns (counting by pitch from the first column split the
    # left half across two blocks, Nov 6 smoke run), so a block of 16 is one whole half.
    xs = sorted({round(x, 6) for x, _ in lay.grid()})
    ys = sorted({round(y, 6) for _, y in lay.grid()})
    for cap in caps:
        u, v = _to_local(cell, cap.xyz[0], cap.xyz[1])
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - u))
        j = min(range(len(ys)), key=lambda k: abs(ys[k] - v))
        keyed.setdefault((i // side, j // side), []).append(cap)
    return [keyed[k] for k in sorted(keyed)]


# ----- asking -----


def state_text(ctx: CapsuleContext, qid: str, extra: str = "") -> str:
    lines = {
        "where_needed": "Two soft-gel inspection cells face each other across an aisle. Each has a table with a tray, a rolling cart of trays, two red leaker bins beside the tray and a blue good bin.",
        "leak_block": "A top view of part of a tray of soft-gel capsules lying on parchment paper. A leaking capsule leaves a pool of its amber fill on the paper around it.",
        "leak_capsule": "A close top view of soft-gel capsules on parchment paper, centred on one capsule. A leaking capsule leaves a pool of its amber fill on the paper around it.",
        "leaker_landed": "A top view of a red leaker bin beside the tray, before and after the robot let go of a capsule over it.",
        "tray_clear": "A top view of a whole tray of soft-gel capsules on parchment after the robot removed the leakers it found. A leaking capsule leaves a pool of its amber fill on the paper around it.",
    }
    return lines[qid] + (" " + extra if extra else "")


def ask(sim: Simulator, ctx: CapsuleContext, qid: str, rgb: np.ndarray | None, truth: str | None, context: dict[str, Any] | None = None, extra: str = "") -> Decision:
    question = QUESTIONS[qid]
    ctx.truth_now = truth
    frame_path = ctx.logbook.save_frame(rgb, qid) if (ctx.logbook and rgb is not None) else None
    decision = ctx.ladder.decide(question, state_text(ctx, qid, extra), encode_png_b64(rgb) if rgb is not None else None, lambda q: {o: True for o in q.options},
                                 t_sim=sim.time, frame_path=frame_path, context=context or {})
    decision.truth = truth
    ctx.decisions.append(decision)
    if ctx.logbook:
        ctx.logbook.record(decision)
    if ctx.on_event:
        ctx.on_event(f"asked_{qid}_{decision.answer}")
    return decision


def default_see(sim: Simulator, ctx: CapsuleContext) -> tuple[np.ndarray | None, list[SeenCapsule]]:
    """The tray camera's picture and the capsules perception finds in its point cloud."""
    cell = ctx.cells.cell(ctx.cell or "a")
    lay = ctx.cells.layout
    rgb, xyz = sim.capture_cloud(f"{cell.name}_tray")
    return rgb, find_capsules(xyz, lay.paper_top, cell.to_world(*lay.tray_center), lay.paper_half, cell.yaw_deg)


# ----- the tree -----


def build_tree(sim: Simulator, ctx: CapsuleContext) -> Sequence:
    cells, lay = ctx.cells, ctx.cells.layout
    see = ctx.see or default_see

    def where(bb: dict) -> bool:
        a, b = sim.capture("a_overview").rgb, sim.capture("b_overview").rgb
        rgb = side_by_side(a, b) if a is not None and b is not None else None
        truth = {"a": "left", "b": "right", None: "neither"}[ctx.episode.work_cell]  # cell a is the left picture
        answer = ask(sim, ctx, "where_needed", rgb, truth, {"work_cell": ctx.episode.work_cell}).answer
        bb["go"] = {"left": "a", "right": "b"}.get(answer)
        return True

    def going(bb: dict) -> bool:
        return bb.get("go") is not None

    def drive(bb: dict) -> bool:
        ctx.cell = bb["go"]
        result = park_station(sim, cells.cell(ctx.cell))
        ctx.parked = result.ok
        p = sim.base_pose()
        ctx.base_home = (float(p.pos[0]), float(p.pos[1]), yaw_of(p.quat))
        bb["note"] = result.reason
        return result.ok

    def scan(bb: dict) -> bool:
        """Blocks first, then each capsule of a block that may hold a leaker. Fills bb['targets'] with world xy."""
        rgb, caps = see(sim, ctx)
        bb["targets"] = []
        if rgb is None or not caps:
            return True
        px_per_m = pixels_per_metre(caps, lay.capsule[0])
        owners = leaker_owners(sim, ctx, caps)

        def label(c: SeenCapsule) -> dict[str, Any]:
            """Sim truth for this capsule, so pictures of any block size can be cut and labelled from the log later."""
            name = owners.get(caps.index(c))
            k = leaker_slot(ctx, name)
            return {"xyz": [round(v, 4) for v in c.xyz], "box": list(c.box), "leaker": name is not None,
                    **({"pool_size": ctx.episode.pool_sizes[k]} if k is not None else {})}

        tray_context = {"capsules": [label(c) for c in caps], "cell": ctx.cell, "block": ctx.block, "px_per_m": round(px_per_m, 1)}
        if ctx.logbook:
            tray_context["tray_frame"] = ctx.logbook.save_frame(rgb, "tray")
        for n, group in enumerate(blocks(caps, ctx)):
            window = padded(rgb, block_box(group), int(round(0.015 * px_per_m)))
            # The pad that keeps an edge capsule's pool in view also takes in the next block's edge capsules, so the
            # truth is over everything the picture shows, not the block alone (commit review round 4, Nov 4).
            shown = in_view(caps, window)
            truth = block_truth(caps, window, set(owners))  # None (a leaker only partly in view): the oracle cannot say, so the ladder's safe answer, yes, looks closer
            picture = crop(rgb, block_box(group), int(round(0.015 * px_per_m)))
            context = {"capsules": [caps.index(c) for c in group], "in_view": [caps.index(c) for c in shown], "window": list(window),
                       **({"tray": tray_context} if n == 0 else {})}  # the whole tray once a scan
            decision = ask(sim, ctx, "leak_block", picture, truth, context)
            if decision.answer != "yes":
                continue
            for cap in group:
                truth = "leaker" if caps.index(cap) in owners else "good"
                decision = ask(sim, ctx, "leak_capsule", crop(rgb, capsule_box(cap, rgb, px_per_m), 0), truth, {"capsule": caps.index(cap), "xyz": [round(v, 4) for v in cap.xyz]})
                if decision.answer == "leaker":
                    bb["targets"].append((cap.xyz[0], cap.xyz[1]))
        return True

    def check_drift() -> None:
        if ctx.base_home is None:
            return
        metres, degrees = base_drift(sim, ctx.base_home)
        mm = metres * 1000
        if (mm > ctx.base_drift_mm + 2.0 or degrees > ctx.base_drift_deg + 0.5) and ctx.on_event:
            ctx.on_event(f"base_drift {mm:.0f} mm, {degrees:.1f} deg since parking")
        ctx.base_drift_mm, ctx.base_drift_deg = max(ctx.base_drift_mm, mm), max(ctx.base_drift_deg, degrees)

    def remove(bb: dict) -> bool:
        cell = cells.cell(ctx.cell)
        for xy in bb.get("targets", []):
            near = leaker_near(sim, ctx, xy)
            name = near if near in cells.leaker_names else loosen(xy)  # a loose leaker back on the paper, or swap one in
            if name is None:
                continue
            arm = side_of(_to_local(cell, float(xy[0]), float(xy[1]))[1])
            picked = pick_capsule(sim, cell, name, lay.paper_top + lay.capsule[2], arm)
            check_drift()  # after every pick, failed or not (review of fdfd388: a failed pick skipped it)
            if not picked.ok:
                ctx.move_failures.append({"move": "pick", "object": name, "arm": arm, "reason": picked.reason})
                if ctx.on_event:
                    ctx.on_event(f"pick_failed {name} {arm}: {picked.reason}")
                travel_pose(sim)
                continue
            camera = f"{cell.name}_leaker_bin_{arm}"
            before = sim.capture(camera).rgb
            dropped = drop_in_bin(sim, cell, name, arm)
            check_drift()
            if not dropped.ok:
                ctx.move_failures.append({"move": "drop", "object": name, "arm": arm, "reason": dropped.reason})
                if ctx.on_event:
                    ctx.on_event(f"drop_failed {name} {arm}: {dropped.reason}")
            travel_pose(sim)  # the hand out of the picture before the after shot
            after = sim.capture(camera).rgb
            ctx.removed.append(name)
            rgb = side_by_side(before, after) if before is not None and after is not None else None
            pos = [round(float(v), 4) for v in sim.object_state(name).pos]
            ask(sim, ctx, "leaker_landed", rgb, "yes" if in_leaker_bin(sim, ctx, name) else "no", {"object": name, "pos": pos})
        return True

    def loosen(xy) -> str | None:
        """The robot is about to pick a capsule the simulator holds fixed: swap in a loose body at the same pose, unseen.
        A leaker spot's capsule becomes that leaker's own body (so the truth follows it); any other becomes a spare,
        counted as a good capsule thrown out."""
        names = [n for n in cells.grid_names if on_paper(ctx, sim.object_state(n).pos)]
        if not names:
            return None
        dists = [float(np.hypot(sim.object_state(n).pos[0] - xy[0], sim.object_state(n).pos[1] - xy[1])) for n in names]
        fixed = names[int(np.argmin(dists))]
        if min(dists) > PICK_MATCH_M:
            return None
        spot = cells.grid_names.index(fixed)
        if spot in ctx.episode.leakers:
            body = cells.leaker_names[ctx.episode.leakers.index(spot)]
        elif ctx.spares_used < len(cells.spare_names):
            body = cells.spare_names[ctx.spares_used]
            ctx.spares_used += 1
        else:
            return None
        state = sim.object_state(fixed)
        sim.set_object_pose(fixed, parked_pos(float(state.pos[0]), float(state.pos[1]), 0.0), 0.0)
        sim.set_object_pose(body, tuple(float(v) for v in state.pos), float(np.degrees(yaw_of(state.quat))))
        ctx.loosened.append(body)
        return body

    def clear(bb: dict) -> bool:
        rgb, _ = see(sim, ctx)
        truth = "no" if ctx.cell == ctx.episode.work_cell and leakers_on_tray(sim, ctx) else "yes"  # the tray it is looking at
        return ask(sim, ctx, "tray_clear", rgb, truth).answer == "yes"

    def leave(bb: dict) -> bool:
        travel_pose(sim)
        if capsule_skills.LOOK_AT_WORK:
            sim.set_head(0.0, 0.0)  # eyes front to back out
        sim.brake(False)
        return back_out(sim).ok

    work = Sequence(
        "work the cell",
        Action("drive there", drive),
        Action("scan the tray", scan),
        Action("remove the leakers", remove),
        Selector("tray clear?", Condition("ask tray_clear == yes", clear), Sequence("scan again", Action("scan the tray", scan), Action("remove the leakers", remove), Action("ask tray_clear", clear))),
        Action("back out", leave),
    )
    # A shift that fails part way still backs out from under the table, and still reports the failure.
    bail = Sequence("back out after a failure", Action("back out", leave), Condition("report the failure", lambda bb: False))
    go = Sequence("go", Condition("a cell needs me", going), Selector("work, or back out", work, bail))
    return Sequence("one shift", Action("ask where_needed", where), Selector("go or stay", go, Condition("nowhere to go", lambda bb: not going(bb))))


@dataclass
class CapsuleResult:
    status: Status
    trace: Trace
    decisions: list[Decision]
    checks: dict[str, Any]
    sim_time: float


def run_capsules(sim: Simulator, ctx: CapsuleContext) -> CapsuleResult:
    trace = Trace()
    status = build_tree(sim, ctx).tick({}, trace)
    left = leakers_on_tray(sim, ctx)
    leakers = set(ctx.cells.leaker_names[: len(ctx.episode.leakers)])
    # A leaker the robot took off the tray but that is in no bin (a drop off the wall, one knocked off by a missed pick)
    # is lost on the floor or the table: the shift did not do its job, whatever tray_clear saw (Nov 5, seed 1001).
    loose = [{"object": n, "pos": [round(float(v), 4) for v in sim.object_state(n).pos]}
             for n in dict.fromkeys(ctx.loosened + ctx.removed) if n in leakers and n not in left and not in_leaker_bin(sim, ctx, n)]
    if loose:
        if ctx.on_event:
            ctx.on_event(f"leakers_loose {loose}")
        status = Status.FAILURE
    checks = {
        "work_cell": ctx.episode.work_cell,
        "went_to": ctx.cell,
        "leakers": len(ctx.episode.leakers),
        "leakers_left_on_tray": len(left),
        # each body once: a leaker that bounced off a bin's rim back onto the tray is picked and dropped twice (seed 1003)
        "leakers_in_bin": sum(1 for n in dict.fromkeys(ctx.removed) if n in leakers and in_leaker_bin(sim, ctx, n)),
        "good_capsules_thrown_out": sum(1 for n in dict.fromkeys(ctx.removed) if n.startswith("spare_")),
        "move_failures": list(ctx.move_failures),
        "base_drift_mm": round(ctx.base_drift_mm, 1),
        "base_drift_deg": round(ctx.base_drift_deg, 2),
        "leakers_loose": loose,
    }
    return CapsuleResult(status, trace, list(ctx.decisions), checks, sim.time)
