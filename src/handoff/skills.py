"""The robot's skills. Each is a callable ``(sim, ctx) -> SkillResult`` registered under a name.

A skill is replaceable: a learned grasp policy registers under ``"grasp"`` with the same
signature and the task never knows. Skills read the world through the simulator interface only.

How the side grasp works (left hand, palm facing the robot's midline, fingers pointing forward):
the wrist is placed beside the part so the part sits against the palm, deep enough that the
fingers wrap around its front and far faces when they curl, and the thumb comes down on top.
The offsets below are in the world frame for the left hand at its home orientation and were
tuned on the measured hand (fingers 0.19 m long, curling toward -y).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .checks import WorldState, grasp_held, in_bin, lifted, moved_with_hand, part_near_wrist, snapshot, still
from .scene import BinRegion, Layout, tray_box_names
from .sim.base import G0, Arm, FingertipSpec, Simulator, quat_from_yaw, quat_mul, yaw_of

# Measured at the kit's home pose (probe, Oct 11); the arm keeps this orientation for every move.
HOME_QUAT = {
    "left": np.array([0.5109, 0.4824, 0.4890, 0.5169]) / np.linalg.norm([0.5109, 0.4824, 0.4890, 0.5169]),
    "right": np.array([-0.506, -0.488, -0.494, -0.512]) / np.linalg.norm([-0.506, -0.488, -0.494, -0.512]),
}
ARM_HOME = {"left": {"left_4": 0.75}, "right": {"right_4": 0.65}}  # the kit home, per arm
TRAVEL_POSE = {"left_4": 1.4, "right_4": 1.4}  # forearms up, hands clear of the table
HEAD_DOWN = 0.6  # rad, positive pitches the head down
HEAD_HEIGHT = 1.40  # m, the head camera above the floor
HEAD_YAW_MAX, HEAD_PITCH_MAX = 1.2, 1.0  # rad
# Crossing the table (Oct 21, dup: the hand bumped the tray walls carrying parts). Every move between spots goes up to one
# clearance height where the hand is, crosses at that height, and only then comes straight down (travel below).
TRAVEL_MARGIN = 0.08  # m, wrist above the highest wall top (plus the held part's lower half): the hand hangs about 3 cm below the wrist
LATERAL_MIN = 0.01  # m, a move with less horizontal travel than this counts as straight up or down
BUMP_RISE = 0.05  # m, how far the hand backs up after touching the table or a tray, and how much higher the retry goes
BUMP_EXTRA_STEP, BUMP_EXTRA_CAP = 0.03, 0.15  # m, how much each bump raises every later crossing, and the most it adds
BUMP_RETRIES = 3  # retries of one move before the skill gives up and reports the miss
HOME_WRIST_XY = {"left": (0.32, 0.22), "right": (0.32, -0.22)}  # where the kit home puts each wrist (probe, Oct 11)
# The hand in plan around its wrist: x from 3 cm behind to 20 cm ahead (fingers 19 cm along +x), y 5 cm either side.
HAND_PLAN = (-0.03, 0.20, 0.05)
TRAY_MARGIN = 0.01  # m kept between the hand's plan and a tray's outer walls when picking the spot to go home from


def bin_seen(ctx: TaskContext, name: str) -> BinRegion:
    """The bin as the robot believes it is: its size from the drawing, its position from the camera when it has one."""
    region = ctx.layout.bin_region(name)
    if name not in ctx.bins_seen:
        return region
    cx, cy = ctx.bins_seen[name]
    return BinRegion(name, cx, cy, region.half_inner, region.floor_z, region.wall_top_z)


def look_at(sim: Simulator, point) -> None:
    """Turn the head so its camera faces a world point: the robot watches what it is working on."""
    base = sim.base_pose()
    dx, dy = float(point[0]) - float(base.pos[0]), float(point[1]) - float(base.pos[1])
    left = np.arctan2(dy, dx) - yaw_of(base.quat)  # bearing to the point, positive to the robot's left
    left = (left + np.pi) % (2 * np.pi) - np.pi
    yaw = -left  # the head_yaw joint turns the head to the robot's RIGHT for positive values (checked on head frames)
    pitch = np.arctan2(HEAD_HEIGHT - float(point[2]), np.hypot(dx, dy))
    sim.set_head(float(np.clip(yaw, -HEAD_YAW_MAX, HEAD_YAW_MAX)), float(np.clip(pitch, 0.0, HEAD_PITCH_MAX)))


@dataclass
class TaskContext:
    layout: Layout = field(default_factory=Layout)
    target_part: str = "part_00"
    parts: tuple[str, ...] = ()  # the parts in play this episode (default: just the target)
    target_bin: str = "good"
    arm: Arm = "left"
    grasp_mode: str = "fingers"  # "fingers" closes the hand; "weld" closes it and welds on contact
    # Wrist position relative to the part centre at the grasp (x: how deep in the palm, y: palm side).
    grasp_offset: tuple[float, float, float] = (-0.12, 0.04, 0.0)  # centre of the region that held in the Oct 11 sweeps
    approach_offset: tuple[float, float, float] = (0.0, 0.03, 0.10)  # come in from above, clear of bin walls
    align_yaw: bool = True  # turn the hand to match the part's yaw (a square part repeats every 90 degrees)
    max_align_deg: float = 20.0  # the wrist joints allow about 17 degrees; beyond that the IK misses by centimetres
    # How the hand closes: "fixed" is the close straight to close_fraction (a position target inside the part, so the
    # fingers sink in); "force" ramps the closure until the fingertip sensors report the grip force and a GripKeeper
    # holds it (grasp_force). "fixed" stays the default until scripts/probe_grip.py on the GPU shows the force grip
    # holding at least as well (its first version dropped the part on lift in the review's CPU run, Oct 21).
    grasp_control: str = "fixed"
    grip: str | None = None  # the --grip choice by name (grip_settings), for the run log; None: named by grasp_control
    grip_force_n: float | None = None  # the finger side's target force (N); None: from the part's mass (grip_target_n)
    grip_timeout_s: float = 3.0  # the force ramp gives up after this long without reaching the grip force
    grip_lead: float = 0.03  # GRIP_MAX_LEAD: how far (closure) the command may run ahead of the measured fingers
    # The palm must feel at least this (N, tactile skin on the gripper base) before the grip counts: the part clamped
    # against the palm, not just touched by the fingers (Genesis CPU, Oct 21: the fixed grasp that holds presses the
    # palm; the force grip that dropped did not). None: not required; also skipped on a hand without skin.
    grip_palm_n: float | None = None
    # After the fingers reach the grip force, close the thumb on its own until its tip feels this much (N). The fixed
    # grasp that holds has the thumb tip at 10 to 14 N; every force grip that dropped had it at 0, because the shared
    # closure reaches the thumb only near 0.85, where the fingers already crush the part (Genesis CPU, Oct 21).
    # None: the thumb follows the fingers (set_hand), as before.
    grip_thumb_n: float | None = None
    close_fraction: float = 0.9  # "fixed" only
    squeeze_effort: float | None = None  # "fixed" only: N*m on the finger roots after closing, None = position hold
    lift_height: float = 0.15
    # Where the robot saw each bin (perception.locate_bins). Empty: it falls back to where the layout says.
    bins_seen: dict[str, tuple[float, float]] = field(default_factory=dict)
    hover_above_bin: float = 0.08  # wrist height above the bin wall top on the way in, clear of the walls and the parts already there
    # Wrist height above the wall top when letting go. The hand is 19 cm long and the bin 16 cm inside, so the hand
    # never fits in the bin: it straddles a wall, and what hangs below the wrist touches the wall top at 2.7 to 3.5 cm
    # (measured Oct 13, win_lower_diag.py, seed 204, all three spots). 4 cm clears that; the part then falls 7 to 8 cm
    # instead of 11 (VALIDATION.md: fewer parts end up lying flat, 6 of 18 against 9, but it is still a drop, not a placement).
    release_above_wall: float = 0.04
    place_slot: int = 0  # which spot in the bin: parts are released at different spots so they do not stack
    move_timeout_s: float = 3.0
    # Opening at once flicks the part 8 to 9 cm away from the robot (measured Oct 12: over the far wall from the third
    # spot, 6 times in 6). Opened over a second it stays within about 2 cm there.
    release_open_s: float = 1.0
    retract_clear_s: float = 1.2  # wait after a release before the in-bin question: at 0.6 s the hand still covered part of the bin in the camera frame
    on_event: Callable[[str], None] | None = None  # e.g. save a camera frame
    travel_margin: float = TRAVEL_MARGIN
    # Every time the hand touched the table or a tray this run, as {"box", "skill", "wrist_z", "retry_z", "clearance_extra"}.
    # The sort hands the same list to every part's context, so what one part bumped raises the crossings of the rest.
    bumps: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def parts_in_play(self) -> tuple[str, ...]:
        return self.parts or (self.target_part,)

    @property
    def clearance_extra(self) -> float:
        """What the bumps so far add to the crossing height: BUMP_EXTRA_STEP each, at most BUMP_EXTRA_CAP."""
        return min(BUMP_EXTRA_CAP, BUMP_EXTRA_STEP * len(self.bumps))

    def event(self, tag: str) -> None:
        if self.on_event is not None:
            self.on_event(tag)


@dataclass
class SkillResult:
    ok: bool
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)


SkillFn = Callable[[Simulator, TaskContext], SkillResult]
SKILLS: dict[str, SkillFn] = {}


def skill(name: str) -> Callable[[SkillFn], SkillFn]:
    def register(fn: SkillFn) -> SkillFn:
        SKILLS[name] = fn
        return fn

    return register


def part_yaw(sim: Simulator, ctx: TaskContext) -> float:
    """The part's heading folded into (-45, 45] degrees (a square part repeats every 90), clamped to what the hand can turn."""
    yaw = np.degrees(yaw_of(sim.object_state(ctx.target_part).quat))
    folded = float((yaw + 45.0) % 90.0 - 45.0)
    return float(np.clip(folded, -ctx.max_align_deg, ctx.max_align_deg))


def grasp_quat(sim: Simulator, ctx: TaskContext) -> np.ndarray:
    """Hand orientation for this grasp: the home orientation, yawed with the part when aligning."""
    if "grasp_quat" in ctx.data:
        return ctx.data["grasp_quat"]
    quat = HOME_QUAT[ctx.arm]
    if ctx.align_yaw:
        quat = quat_mul(quat_from_yaw(np.radians(part_yaw(sim, ctx))), quat)
    ctx.data["grasp_quat"] = quat
    return quat


def grasp_wrist_pos(sim: Simulator, ctx: TaskContext) -> np.ndarray:
    """Wrist position for the grasp: the offset, rotated with the part when aligning."""
    part = sim.object_state(ctx.target_part).pos
    offset = np.asarray(ctx.grasp_offset, dtype=float)
    if ctx.align_yaw:
        yaw = np.radians(part_yaw(sim, ctx))
        c, s = np.cos(yaw), np.sin(yaw)
        offset = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]])
    return part + offset


def fixed_boxes(ctx: TaskContext) -> tuple[str, ...]:
    """What the hand must not touch on the way between spots: the table and every piece of every tray."""
    return ("table", *(name for b in ctx.layout.bin_names for name in tray_box_names(b)))


def part_in_hand(state: WorldState, ctx: TaskContext) -> bool:
    """The hand still has the part: held by the fingers (grasp_held: near the wrist, touched, off the table), or, in
    weld mode, welded to it, which needs no finger contact: the touch grip rests its fingers at the surface, where the
    contact solver can read nothing while the weld carries the part (3 of 30 touch lifts read "dropped", Oct 21)."""
    part = ctx.target_part
    if ctx.grasp_mode == "weld":
        return part_near_wrist(state, part, ctx.arm) and lifted(state, part)
    return grasp_held(state, part, ctx.arm)


def holding(sim: Simulator, ctx: TaskContext) -> bool:
    """The hand has the part, or might: closed (the finger joints' encoders), or a fingertip still feels something (a
    part that sticks after a release counts). A hand without fingertip sensors falls back to the contact solver, which
    is truth, not a sensor (base.py)."""
    if sim.hand_closure(ctx.arm) > 0.5:
        return True
    tips = sim.read_fingertips(ctx.arm)
    if tips:
        return any(v > 0.0 for v in tips.values())
    return sim.hand_contacts(ctx.arm, ctx.target_part) > 0


def clearance_z(sim: Simulator, ctx: TaskContext) -> float:
    """The wrist height every crossing happens at: the highest obstacle (tray wall tops, the table), the held part's
    lower half when the hand has one, the margin, and whatever the bumps so far have added."""
    top = max([ctx.layout.table_top, *(bin_seen(ctx, b).wall_top_z for b in ctx.layout.bin_names)])
    held = ctx.layout.part_size[2] / 2.0 if holding(sim, ctx) else 0.0
    return top + held + ctx.travel_margin + ctx.clearance_extra


def report_bump(sim: Simulator, ctx: TaskContext, skill_name: str, retry_z: float | None) -> None:
    """Put the bump the last move stopped on in ``ctx.bumps`` (which raises later crossings by BUMP_EXTRA_STEP) and
    out as a ``bump_<box>`` event, which the demo writes to the run log. ``retry_z`` is None when the move is not tried
    again (``lower`` lets go where it stopped)."""
    here = sim.wrist_pose(ctx.arm).pos
    extra = min(BUMP_EXTRA_CAP, BUMP_EXTRA_STEP * (len(ctx.bumps) + 1))
    ctx.bumps.append({"box": sim.last_bump, "skill": skill_name, "wrist_z": round(float(here[2]), 4), "retry_z": None if retry_z is None else round(retry_z, 4), "clearance_extra": round(extra, 4)})
    ctx.event(f"bump_{sim.last_bump}")


def back_off(sim: Simulator, ctx: TaskContext, skill_name: str, retry_z: float, quat: np.ndarray) -> float:
    """The last move stopped on a bump: report it, back the hand straight up (at least to ``retry_z``, the height the
    move is tried again at) and return ``retry_z``."""
    report_bump(sim, ctx, skill_name, retry_z)
    here = sim.wrist_pose(ctx.arm).pos
    up = max(float(here[2]) + BUMP_RISE, retry_z)
    sim.move_wrist(ctx.arm, np.array([here[0], here[1], up]), quat, timeout_s=1.5, tol_m=0.02, free_yaw=True)
    return retry_z


def travel(sim: Simulator, ctx: TaskContext, target: np.ndarray, skill_name: str, *, timeout_s: float, tol_m: float) -> tuple[float, float]:
    """Take the wrist to ``target`` without sweeping anything: up to the clearance height where it is (never down),
    across at that height, then straight down. Each leg is a straight line with the hand kept level and free to turn
    about vertical (``free_yaw``: holding the grasp yaw too drove the wrist into its joint stops mid-crossing).

    The crossing and the descent stop if the hand touches the table or a tray; the hand then backs up and the move is
    tried again BUMP_RISE higher (a descent's end point too), up to BUMP_RETRIES times. Returns the final distance
    from the target, as raised by any retries, and the height the last crossing was made at.
    """
    arm, quat, boxes = ctx.arm, grasp_quat(sim, ctx), fixed_boxes(ctx)
    goal = np.asarray(target, dtype=float).copy()
    here = sim.wrist_pose(arm).pos
    z = max(float(here[2]), clearance_z(sim, ctx))
    if z - float(here[2]) > 0.005:
        sim.move_wrist(arm, np.array([here[0], here[1], z]), quat, timeout_s=2.0, tol_m=0.02, free_yaw=True)
    for _ in range(BUMP_RETRIES + 1):
        if float(np.linalg.norm(sim.wrist_pose(arm).pos[:2] - goal[:2])) > LATERAL_MIN:
            sim.move_wrist(arm, np.array([goal[0], goal[1], z]), quat, timeout_s=timeout_s, tol_m=0.02, stop_on=boxes, free_yaw=True)
            if sim.last_bump is not None:
                z = back_off(sim, ctx, skill_name, z + BUMP_RISE, quat)
                continue
        err = float(np.linalg.norm(sim.wrist_pose(arm).pos - goal))
        if err <= tol_m:  # the crossing already ended there (a target at the crossing height)
            return err, z
        err = sim.move_wrist(arm, goal, quat, timeout_s=timeout_s, tol_m=tol_m, stop_on=boxes, free_yaw=True)
        if sim.last_bump is None:
            return err, z
        goal[2] += BUMP_RISE
        back_off(sim, ctx, skill_name, float(goal[2]), quat)
    return float(np.linalg.norm(sim.wrist_pose(arm).pos - goal)), z


def _hand_over_tray(ctx: TaskContext, x: float, y: float) -> bool:
    """With the wrist at (x, y), the hand's plan overlaps some tray's outer footprint (plus TRAY_MARGIN)."""
    hx0, hx1, hy = x + HAND_PLAN[0], x + HAND_PLAN[1], HAND_PLAN[2]
    for name in ctx.layout.bin_names:
        region = bin_seen(ctx, name)
        half = region.half_inner + ctx.layout.bin_thick + TRAY_MARGIN
        if hx0 < region.cx + half and hx1 > region.cx - half and y - hy < region.cy + half and y + hy > region.cy - half:
            return True
    return False


def clear_spot(ctx: TaskContext) -> tuple[float, float]:
    """Where the wrist goes before the joint move home: over its home spot, slid toward the midline until the hand
    is clear of both trays (the joint move then comes nearly straight down, sweeping no wall)."""
    x, y = HOME_WRIST_XY[ctx.arm]
    toward_mid = -1.0 if y > 0 else 1.0
    for _ in range(30):
        if not _hand_over_tray(ctx, x, y):
            break
        y += 0.01 * toward_mid
    return x, y


def leave_bin(sim: Simulator, ctx: TaskContext, skill_name: str) -> float:
    """Up to the clearance height, across to the clear spot, at that height: the open hand never sweeps a wall."""
    x, y = clear_spot(ctx)
    z = max(float(sim.wrist_pose(ctx.arm).pos[2]), clearance_z(sim, ctx))
    return travel(sim, ctx, np.array([x, y, z]), skill_name, timeout_s=ctx.move_timeout_s, tol_m=0.02)[0]


@skill("drive_to_table")
def drive_to_table(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """Drive forward until the base origin reaches the park x, holding heading and y near zero."""
    target_x = ctx.layout.park_x
    sim.set_joints(TRAVEL_POSE)
    sim.set_head(0.0, HEAD_DOWN)
    max_steps = int(20.0 / sim.dt)
    done = 0
    while done < max_steps:
        pose = sim.base_pose()
        dx = target_x - float(pose.pos[0])
        if dx <= 0.01:
            break
        yaw = yaw_of(pose.quat)
        linear = float(np.clip(dx * 1.0, 0.08, 0.4))
        angular = float(np.clip(-1.5 * yaw - 0.8 * float(pose.pos[1]), -0.5, 0.5))
        sim.drive(linear, angular)
        sim.step(5)
        done += 5
    sim.stop()
    sim.settle(0.5)
    pose = sim.base_pose()
    err_x, err_y, err_yaw = float(pose.pos[0]) - target_x, float(pose.pos[1]), yaw_of(pose.quat)
    ok = abs(err_x) <= 0.05 and abs(err_y) <= 0.05 and abs(err_yaw) <= 0.1
    return SkillResult(ok, "" if ok else "parked off target", {"x": err_x, "y": err_y, "yaw": err_yaw})


@skill("look_at_table")
def look_at_table(sim: Simulator, ctx: TaskContext) -> SkillResult:
    sim.set_head(0.0, HEAD_DOWN)
    sim.settle(0.5)
    return SkillResult(True)


@skill("reach")
def reach(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """Arm to the pre-grasp pose beside the part, hand open."""
    sim.set_joints({f"{ctx.arm}_{i}": 0.0 for i in range(1, 8)} | ARM_HOME[ctx.arm])
    let_go(sim, ctx.arm)
    ctx.data.pop("grasp_quat", None)  # re-read the part's yaw for this attempt
    pregrasp = grasp_wrist_pos(sim, ctx) + np.asarray(ctx.approach_offset)
    err = sim.move_wrist(ctx.arm, pregrasp, grasp_quat(sim, ctx), timeout_s=ctx.move_timeout_s, tol_m=0.01)
    ok = err <= 0.02
    return SkillResult(ok, "" if ok else f"pre-grasp pose missed by {err:.3f} m", {"err": err})


@skill("approach")
def approach(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """Bring the open hand straight down beside the part, then slide it in so the part sits against the palm.

    One diagonal move from the pre-grasp pose cut the corner; down first, then the short slide, keeps the palm off
    the part on the way down."""
    target = grasp_wrist_pos(sim, ctx)
    quat = grasp_quat(sim, ctx)
    beside = target + np.array([ctx.approach_offset[0], ctx.approach_offset[1], 0.0])
    if float(np.linalg.norm(beside - target)) > 1e-3:
        sim.move_wrist(ctx.arm, beside, quat, timeout_s=ctx.move_timeout_s, tol_m=0.01)
    err = sim.move_wrist(ctx.arm, target, quat, timeout_s=ctx.move_timeout_s, tol_m=0.01)
    ok = err <= 0.02
    return SkillResult(ok, "" if ok else f"grasp pose missed by {err:.3f} m", {"err": err})


# ----- the grip: close until the fingertips feel the part, not to a position inside it -----
#
# Closing to close_fraction 0.9 commands a finger position inside the 4 cm part, so the PD fingers drive into it until
# the contact solver pushes back: they visibly sink in (dup saw it in the demo, Oct 21). A real hand with fingertip
# force sensors closes until the force is there and holds it. The force target follows from the part (grip_budget):
#
# - Friction mu = 1.5. Genesis gives a contact the larger of its two geoms' frictions (genesis/engine/solvers/rigid/
#   collider/contact.py, ``friction = max(max(friction_a, friction_b), 1e-2)``). The part is friction 1.5
#   (scene.part_boxes); the hand's collision prims bind no physics material in assets/MH3.usd (probe Oct 21: every
#   L_* collision prim, physics binding None), so they get Genesis's default 1.0 (genesis/utils/geom.py
#   ``default_friction``). The table's BoxSpec default is 1.0. So every contact the part makes is 1.5.
# - Hold: the part is squeezed between two opposing sides (palm against the finger tips on its far face), each with
#   its own friction, so the finger side needs N >= SF * m * g / (2 * mu). The 4 x 4 x 8 cm part at 400 kg/m^3 is
#   51.2 g, 0.502 N: N >= 2.5 * 0.502 / 3.0 = 0.42 N.
# - Safety factor 2.5. move_wrist eases each move as d * (1 - cos(pi t / T)) / 2 with T = clip(d / 0.25 m/s, 0.3, 2 s),
#   so its peak acceleration pi^2 d / (2 T^2) is largest, 4.1 m/s^2, for a 7.5 cm move at the 0.3 s floor (the
#   15 cm lift: 2.1 m/s^2). Upward that is (9.81 + 4.1) / 9.81 = 1.42 g on the part. The fingertip FSRs read up to 18%
#   low (3 sigma of the datasheet's 6% part-to-part spread): 1.42 / 0.82 = 1.73. The rest, to 2.5, is margin for what
#   this leaves out: the arm's PD tracking overshooting the eased profile and the soft contact's creep.
# - Clamped, not pushed: before the part is against the palm, fingers that push it along the table read its sliding
#   friction, mu * m * g = 0.75 N, which is more than the 0.42 N hold. So the target is also above the most a sliding
#   push can read: 0.75 N times (1 + 3 * (6% part to part + 2% repeatability)) = 0.93 N. Reaching that means the part
#   is clamped against something. This floor is the larger, so the target is 0.93 N on the four finger tips together.
#
# The closure ramps fast until any fingertip feels a touch, then slowly, so the force grows little between two reads:
# at the slow rate one read (2 steps, 0.02 s) advances the fingers 0.005 of their 1.45 rad travel, 7 mrad, which at
# the finger PD stiffness (kp 5 N*m/rad, genesis_sim.HAND_KP) is 0.036 N*m more at the root. On the first read at the
# target the ramp holds still; a second read at the target ends it (one noisy read never does).
#
# Two things the first version got wrong on Genesis (review CPU run, Oct 21, seeds 0 and 1; VALIDATION.md):
# - The command ran up to 0.08 closure ahead of the lagging fingers, so the force overshot to 14 N, 15x the target.
#   The command now never leads the measured closure by more than GRIP_MAX_LEAD: 0.03 closure is 44 mrad of PD
#   error, 0.22 N*m at a finger root, a few newtons at the tip, which is all the press the grip needs.
# - The fingers crept on to closure 0.80 during the hold (the soft contact gives), past the held command 0.74; on the
#   lift they sprang back to the command, every tip read 0 and the part fell. A fixed command cannot hold a part that
#   gives. So once the grip is reached, a GripKeeper (a per-step controller, Simulator.add_controller) holds it the way
#   a hand with fingertip sensors does: every read below the target closes a little more (never more than
#   GRIP_MAX_LEAD past where the fingers are), and fingers that close to EMPTY_CLOSURE without the force mean the part
#   is gone. The keeper runs through the lift and carry and comes off when the hand opens (open_hand).
GRIP_MU = 1.5
GRIP_SURFACES = 2
GRIP_SAFETY = 2.5
GRIP_SIDE = ("index", "middle", "ring", "little")  # the tips that press the part's far face against the palm
GRIP_FAST_RATE, GRIP_SLOW_RATE = 1.0, 0.25  # closure per second, before and after the first touch
GRIP_READ_EVERY = 2  # physics steps between fingertip reads
GRIP_CONFIRM = 2  # consecutive reads at the target before the ramp ends
GRIP_CATCH_UP_S = 0.3  # once commanded fully closed, how long the fingers get to arrive before the ramp gives up
GRIP_HOLD_S = 0.2  # hold with the keeper on before the grip is reported; it must still be at the target after
GRIP_MAX_LEAD = 0.03  # the command never runs more than this (closure) ahead of the measured fingers
GRIP_HOLD_FLOOR = 0.8  # after the hold the finger side must read at least this share of the target (FSR noise is 2%)
# The empty hand presses on itself from here on: ramped at the slow rate with nothing in it, the index and thumb tips
# first touch each other at a measured closure of 0.818 (command 0.845) and read 5.6 to 5.8 N at once (CPU probe,
# Oct 21, one run; a snap close to 0.92 also put the other three tips on the palm). A grip force first reached at or
# past this closure is the hand on itself, not a part. On the 4 cm part the tips first touched at closure 0.676 (review
# CPU run, seeds 0 and 1), so the margin is 0.12; scripts/probe_grip.py records closure_at_grip across seeds.
EMPTY_CLOSURE = 0.80


def grip_budget(layout: Layout, fingertip: FingertipSpec | None = None, mu: float = GRIP_MU, safety: float = GRIP_SAFETY) -> dict[str, float]:
    """The finger side's grip force for this layout's part, and what it is made of (see the notes above)."""
    spec = fingertip or FingertipSpec()
    mass = layout.part_rho * float(np.prod(layout.part_size))
    weight = mass * G0
    hold = safety * weight / (GRIP_SURFACES * mu)
    slide = mu * weight * (1.0 + 3.0 * (spec.part_to_part + spec.repeatability))
    return {"mass_kg": mass, "weight_n": weight, "hold_n": hold, "slide_floor_n": slide, "target_n": max(hold, slide)}


def grip_target_n(layout: Layout, fingertip: FingertipSpec | None = None) -> float:
    return grip_budget(layout, fingertip)["target_n"]


def finger_side_n(tips: dict[str, float], arm: Arm) -> float:
    """The force the four finger tips press with together (the side the palm opposes)."""
    return float(sum(tips.get(f"{arm}_{f}_tip", 0.0) for f in GRIP_SIDE))


class GripKeeper:
    """Holds a reached grip through the lift and carry: a per-step controller (Simulator.add_controller) that reads
    the fingertips every GRIP_READ_EVERY steps and, on any read below the target, closes a little more (at the slow
    rate, never more than GRIP_MAX_LEAD past the measured fingers). Fingers that reach EMPTY_CLOSURE without the force
    mean the part is gone: the keeper marks it lost and stops tightening (it never opens the hand)."""

    def __init__(self, sim: Simulator, arm: Arm, target_n: float, cmd: float, lead: float = GRIP_MAX_LEAD):
        self.sim, self.arm, self.target_n, self.cmd, self.lead = sim, arm, float(target_n), float(cmd), float(lead)
        self.thumb_cmd: float | None = None  # set when the thumb was closed on its own; kept on every re-tighten
        self.steps = self.reads = self.tightened = 0
        self.lost = False
        self.min_n = self.last_n = float("inf")

    def __call__(self) -> None:
        self.steps += 1
        if self.steps % GRIP_READ_EVERY or self.lost:
            return
        side = finger_side_n(self.sim.read_fingertips(self.arm), self.arm)
        self.reads += 1
        self.last_n, self.min_n = side, min(self.min_n, side)
        if side >= self.target_n:
            return
        closure = self.sim.hand_closure(self.arm)
        if closure >= EMPTY_CLOSURE:
            self.lost = True
            return
        cmd = min(1.0, self.cmd + GRIP_SLOW_RATE * self.sim.dt * GRIP_READ_EVERY, closure + self.lead)
        if cmd > self.cmd:
            self.cmd = cmd
            self.sim.set_hand(self.arm, cmd)
            if self.thumb_cmd is not None:
                self.sim.set_thumb(self.arm, self.thumb_cmd)
            self.tightened += 1

    def summary(self) -> dict[str, float | int | bool]:
        return {"lost": self.lost, "tightened": self.tightened, "reads": self.reads, "cmd": self.cmd,
                "min_n": self.min_n if self.reads else None, "last_n": self.last_n if self.reads else None}


def grip_keeper(sim: Simulator, arm: Arm) -> GripKeeper | None:
    keeper = sim.controller(f"grip_keeper:{arm}")
    return keeper if isinstance(keeper, GripKeeper) else None


def palm_n(sim: Simulator, arm: Arm) -> float | None:
    """The force the palm's skin feels (N), or None on a hand without skin there."""
    skin = sim.read_tactile(arm)
    force = skin.get(f"{arm}_gripper_base")
    return None if force is None else float(np.linalg.norm(force))


def palm_ok(sim: Simulator, arm: Arm, need_n: float | None) -> bool:
    if need_n is None:
        return True
    palm = palm_n(sim, arm)
    return palm is None or palm >= need_n


def close_thumb(sim: Simulator, ctx: TaskContext, start: float) -> dict[str, float | bool]:
    """Close the thumb alone from ``start`` (where set_hand left it) until its tip reads ``ctx.grip_thumb_n`` on two
    reads in a row, the command never more than ``ctx.grip_lead`` ahead of the measured thumb; gives up at full closure
    or after ``ctx.grip_timeout_s``. The fingers stay where they are commanded."""
    arm, need = ctx.arm, float(ctx.grip_thumb_n or 0.0)
    tip = f"{arm}_thumb_tip"
    cmd, confirm, steps = float(start), 0, 0
    max_steps = max(1, round(ctx.grip_timeout_s / sim.dt))
    if not sim.set_thumb(arm, cmd):
        return {"separate": False, "cmd": cmd, "tip_n": sim.read_fingertips(arm).get(tip, 0.0), "target_n": need, "reached": False}
    force = sim.read_fingertips(arm).get(tip, 0.0)
    while steps < max_steps:
        confirm = confirm + 1 if force >= need else 0
        if confirm >= GRIP_CONFIRM or (cmd >= 1.0 and sim.thumb_closure(arm) >= 0.99):
            break
        if confirm == 0:
            rate = GRIP_SLOW_RATE if force > 0.0 else GRIP_FAST_RATE
            cmd = min(1.0, cmd + rate * sim.dt * GRIP_READ_EVERY, sim.thumb_closure(arm) + ctx.grip_lead)
            sim.set_thumb(arm, cmd)
        sim.step(GRIP_READ_EVERY)
        steps += GRIP_READ_EVERY
        force = sim.read_fingertips(arm).get(tip, 0.0)
    return {"separate": True, "cmd": cmd, "tip_n": force, "target_n": need, "closure": sim.thumb_closure(arm),
            "reached": confirm >= GRIP_CONFIRM, "s": steps * sim.dt}


def start_grip_keeper(sim: Simulator, arm: Arm, target_n: float, cmd: float, lead: float = GRIP_MAX_LEAD) -> GripKeeper:
    keeper = GripKeeper(sim, arm, target_n, cmd, lead)
    sim.add_controller(f"grip_keeper:{arm}", keeper)
    return keeper


def stop_grip_keeper(sim: Simulator, arm: Arm) -> GripKeeper | None:
    """Take the keeper off (before the hand opens, or it would close again on the falling force); returns it."""
    remove = getattr(sim, "remove_controller", None)  # a bare hand stand-in (tests) has no controllers
    keeper = remove(f"grip_keeper:{arm}") if remove is not None else None
    return keeper if isinstance(keeper, GripKeeper) else None


def grasp_fixed(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """The old grasp, kept for the before/after: close to ``close_fraction`` at once and judge it by the contact
    solver (truth: the force grasp judges by the fingertip sensors instead)."""
    t0 = sim.time
    stop_grip_keeper(sim, ctx.arm)
    sim.set_hand(ctx.arm, ctx.close_fraction)
    sim.settle(1.0)
    if ctx.squeeze_effort is not None:
        sim.squeeze(ctx.arm, ctx.squeeze_effort)
        sim.settle(0.5)
    contacts = sim.hand_contacts(ctx.arm, ctx.target_part)
    closure = sim.hand_closure(ctx.arm)
    tips = sim.read_fingertips(ctx.arm)  # for the log line only
    data = {"control": "fixed", "contacts": contacts, "closure": closure, "closure_cmd": ctx.close_fraction, "tips": tips,
            "finger_side_n": finger_side_n(tips, ctx.arm), "grasp_s": sim.time - t0}
    if contacts == 0:
        return SkillResult(False, "hand closed without touching the part", data)
    return SkillResult(True, "", data)


def grasp_force(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """Ramp the hand closed, reading the fingertips every GRIP_READ_EVERY steps, until the finger tips together press
    with the grip force; then hold the closure there. Judged by the robot's own sensors only: the fingertip FSRs and
    the finger joints' measured closure.

    Fails with "hand closed without touching the part" when the fingers close all the way (or reach the force only
    where the empty hand presses on itself, EMPTY_CLOSURE) and with "never reached grip force" when something is
    touched but the force does not come within ``grip_timeout_s`` or before the hand is fully closed."""
    arm = ctx.arm
    target = ctx.grip_force_n if ctx.grip_force_n is not None else grip_target_n(ctx.layout, sim.sensors.fingertip)
    t0 = sim.time
    cmd = sim.hand_closure(arm)
    tips = sim.read_fingertips(arm)
    stop_grip_keeper(sim, arm)
    touched, at_target, reads, steps, closed_steps = False, 0, 0, 0, 0
    max_steps = max(1, round(ctx.grip_timeout_s / sim.dt))
    catch_up = round(GRIP_CATCH_UP_S / sim.dt)
    while True:
        touched = touched or any(v > 0.0 for v in tips.values())
        at_target = at_target + 1 if finger_side_n(tips, arm) >= target and palm_ok(sim, arm, ctx.grip_palm_n) else 0
        if at_target >= GRIP_CONFIRM or steps >= max_steps:
            break
        if cmd >= 1.0:
            closed_steps += GRIP_READ_EVERY
            if closed_steps > catch_up:
                break
        if at_target == 0:  # at the target once: hold still and read again before stopping
            rate = GRIP_SLOW_RATE if touched else GRIP_FAST_RATE
            cmd = min(1.0, cmd + rate * sim.dt * GRIP_READ_EVERY, sim.hand_closure(arm) + ctx.grip_lead)
            sim.set_hand(arm, cmd)
        sim.step(GRIP_READ_EVERY)
        steps += GRIP_READ_EVERY
        tips = sim.read_fingertips(arm)
        reads += 1
    closure = sim.hand_closure(arm)
    reached = at_target >= GRIP_CONFIRM
    keeper = None
    thumb: dict[str, float | bool] | None = None
    if reached and closure < EMPTY_CLOSURE and ctx.grip_thumb_n is not None:
        thumb = close_thumb(sim, ctx, cmd)
    if reached and closure < EMPTY_CLOSURE:
        keeper = start_grip_keeper(sim, arm, target, cmd, lead=ctx.grip_lead)
        if thumb is not None and thumb["separate"]:
            keeper.thumb_cmd = float(thumb["cmd"])
        sim.settle(GRIP_HOLD_S)  # the keeper holds the force while the contact settles
        tips = sim.read_fingertips(arm)
    side = finger_side_n(tips, arm)
    data = {"control": "force", "target_n": target, "finger_side_n": side, "tips": tips, "closure": sim.hand_closure(arm),
            "closure_at_grip": closure, "closure_cmd": keeper.cmd if keeper else cmd, "touched": touched, "reads": reads,
            "grasp_s": sim.time - t0}
    if keeper is not None:
        data["keeper"] = keeper.summary()
    if thumb is not None:
        data["thumb"] = thumb
    if reached and closure >= EMPTY_CLOSURE:
        return SkillResult(False, f"hand closed without touching the part (the fingers pressed on the hand itself at closure {closure:.2f})", data)
    if keeper is not None and ctx.grasp_mode == "weld" and not keeper.lost:
        # The touch grip: the fingertips felt the part and the hand is not closed on itself; the weld holds it from
        # here, so the light touch is not held to the squeeze's force floor (on seeds 1 to 10 that check refused 23 of
        # 30 touch grips that read 0.4 to 0.7 N after the hold, Oct 21, scripts/probe_carry.py).
        return SkillResult(True, "", data)
    if keeper is not None and (keeper.lost or side < GRIP_HOLD_FLOOR * target):
        stop_grip_keeper(sim, arm)
        return SkillResult(False, f"grip did not hold: {side:.2f} of {target:.2f} N on the finger tips after the {GRIP_HOLD_S:.1f} s hold", data)
    if reached:
        return SkillResult(True, "", data)
    if not touched:
        return SkillResult(False, "hand closed without touching the part", data)
    when = f"after {ctx.grip_timeout_s:.1f} s" if steps >= max_steps else "with the hand fully closed"
    return SkillResult(False, f"never reached grip force: {side:.2f} of {target:.2f} N on the finger tips {when}", data)


stop_grip_keeper_after_weld = True  # module switch for scripts/probe_carry.py --keep-keeper (the A/B)

# The touch grip (--grip touch): close until the fingertips feel the part (grasp_force), close the thumb on its own to
# this much (N) so it rests on the part, and weld the part to the hand. In the sim a squeeze turns into the fingers
# sinking into the part (the contact is soft: 13 to 16 mm at the fixed 0.9); a metal part in a real hand does not
# give, so the weld stands in for that rigid grip and the fingers stop at the surface (Oct 21, dup).
TOUCH_THUMB_N = 0.5


def grip_settings(grip: str) -> dict[str, Any]:
    """TaskContext/SortContext fields for a --grip choice: fixed, force or touch."""
    if grip == "touch":
        return {"grip": grip, "grasp_control": "force", "grasp_mode": "weld", "grip_thumb_n": TOUCH_THUMB_N}
    if grip in ("fixed", "force"):
        return {"grip": grip, "grasp_control": grip}
    raise ValueError(f"grip must be fixed, force or touch, not {grip!r}")


@skill("grasp")
def grasp(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """Close the hand on the part: the fixed closure by default (grasp_fixed), force-controlled with
    ``grasp_control="force"`` when the hand carries fingertip sensors (grasp_force). In weld mode, also weld once it
    holds."""
    if ctx.grasp_control not in ("force", "fixed"):
        raise ValueError(f"grasp_control must be 'force' or 'fixed', not {ctx.grasp_control!r}")
    if ctx.grasp_control == "force" and sim.read_fingertips(ctx.arm):
        result = grasp_force(sim, ctx)
    else:
        result = grasp_fixed(sim, ctx)
        if ctx.grasp_control == "force":
            result.data["control"] = "fixed (no fingertip sensors)"
    result.data["grip"] = ctx.grip or ctx.grasp_control  # what was asked for; the log names the grip by this
    if result.ok and ctx.grasp_mode == "weld":
        sim.attach(ctx.target_part, ctx.arm)
        result.data["welded"] = True
        # The weld holds the part now; a grip keeper squeezing it on would push the fingers against a part that
        # cannot move (the weld stands in for a rigid part's friction grip).
        if stop_grip_keeper_after_weld:
            stop_grip_keeper(sim, ctx.arm)
            # Rest the fingers where they are: a command past them keeps pushing on a part the weld holds still. Meant
            # to quiet the hand IMU's stops on touch carries (16 of 30, Oct 21); not established: one 30-carry run
            # fell from 33 stops to 19, the next was back at 33 (VALIDATION.md).
            sim.set_hand(ctx.arm, sim.hand_closure(ctx.arm))
            if result.data.get("thumb", {}).get("separate"):
                sim.set_thumb(ctx.arm, sim.thumb_closure(ctx.arm))
    return result


def _tip_label(tip: str, arm: str) -> str:
    return tip.removeprefix(f"{arm}_").removesuffix("_tip")


def grip_line(part: str, result: SkillResult, arm: str = "left") -> str:
    """The run log's line for one grasp: the grip asked for (``data["grip"]``, set by the grasp skill from the context,
    never guessed from how it went) and how it closed when that differs, the finger side's force against its target,
    every tip, the thumb when it closed on its own, how long it took, and whether it held."""
    d = result.data
    control, target, welded = d.get("control", "?"), d.get("target_n"), bool(d.get("welded"))
    grip = d.get("grip") or control
    # the touch grip closes by force, so "force" is its usual close; any other close is named after the grip
    name = grip if control in (grip, "force") else f"{grip} as {control}"
    side = f"finger tips {float(d.get('finger_side_n', 0.0)):.2f}"
    if welded and control == "force":  # the weld holds the part, so the squeeze's target is not asked of it
        force = side + (f" N (the {float(target):.2f} N squeeze not needed under the weld)" if target is not None else " N")
    else:  # a force close that failed did need its target once (grasp_force), the touch grip's included
        force = side + (f" of {float(target):.2f} N" if target is not None else " N")
    control = f"{name} (welded)" if welded else name
    tips = ", ".join(f"{_tip_label(t, arm)} {float(v):.2f}" for t, v in (d.get("tips") or {}).items())
    thumb = d.get("thumb") or {}
    if thumb.get("separate"):
        thumb_text = (f", thumb on its own {float(thumb.get('tip_n', 0.0)):.2f} of {float(thumb.get('target_n', 0.0)):.2f} N"
                      + ("" if thumb.get("reached") else " (short)"))
    elif thumb:
        thumb_text = ", thumb closed with the fingers (no separate thumb)"
    else:
        thumb_text = ""
    end = "held" if result.ok else f"FAILED: {result.reason}"
    return (f"grip {part}: {control}, closure {float(d.get('closure', 0.0)):.2f} (command {float(d.get('closure_cmd', 0.0)):.2f}), "
            f"{force} [{tips}]{thumb_text}, {float(d.get('grasp_s', 0.0)):.2f} s, {end}")


def keeper_line(part: str, summary: dict | None) -> str | None:
    """The run log's line for how a force grip held from the grasp to the release (None when no keeper held it)."""
    if not summary:
        return None
    low = summary.get("min_n")
    low_text = f", lowest {float(low):.2f} N" if low is not None else ""
    if summary.get("lost"):
        return f"grip keeper {part}: LOST the part (fingers closed to {EMPTY_CLOSURE:.2f} without the force){low_text}, tightened {summary.get('tightened', 0)} times"
    return f"grip keeper {part}: held to the release{low_text}, tightened {summary.get('tightened', 0)} times, command {float(summary.get('cmd', 0.0)):.2f}"


def grip_rows(tips: dict[str, float], target_n: float, closure: float, arm: str = "left", tip_max: float = FingertipSpec().max_force,
              keeper: GripKeeper | None = None) -> list[dict[str, str]]:
    """The "Grip force" sensor window: the finger side against its target, the hand's closure, the grip keeper when one
    is holding, then each tip."""
    side = finger_side_n(tips, arm)  # type: ignore[arg-type]
    rows = [
        {"label": "finger side", "value": f"{side:.2f} N", "detail": f"target {target_n:.2f} N (index to little)",
         "state": "ok" if side >= target_n else ("idle" if side <= 0.0 else "warn")},
        {"label": "closure", "value": f"{closure:.2f}", "detail": "finger joints, 0 open to 1 closed", "state": "ok" if closure > 0.05 else "idle"},
    ]
    if keeper is not None:
        rows.append({"label": "grip keeper", "value": "LOST" if keeper.lost else "holding",
                     "detail": f"tightened {keeper.tightened} times, command {keeper.cmd:.2f}", "state": "warn" if keeper.lost else "ok"})
    for tip, force in tips.items():
        name = _tip_label(tip, arm)
        role = "presses the far face" if name in GRIP_SIDE else "on top, not in the grip sum"
        rows.append({"label": name, "value": f"{force:.2f} N", "detail": f"at its {tip_max:.0f} N range" if force >= tip_max else role,
                     "state": "warn" if force >= tip_max else ("ok" if force > 0.0 else "idle")})
    return rows


@skill("lift")
def lift(sim: Simulator, ctx: TaskContext) -> SkillResult:
    before = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    target = sim.wrist_pose(ctx.arm).pos + np.array([0.0, 0.0, ctx.lift_height])
    err = sim.move_wrist(ctx.arm, target, grasp_quat(sim, ctx), timeout_s=ctx.move_timeout_s, tol_m=0.01)
    after = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    held = part_in_hand(after, ctx)
    followed = moved_with_hand(before, after, ctx.target_part, ctx.arm)
    data = {"err": err, "held": held, "followed": followed}
    if not held:
        return SkillResult(False, "part did not come up with the hand", data)
    return SkillResult(True, "", data)


@skill("move_to_bin")
def move_to_bin(sim: Simulator, ctx: TaskContext) -> SkillResult:
    region = bin_seen(ctx, ctx.target_bin)
    # The part hangs at wrist - grasp_offset, so aim the wrist so the part is over the bin centre.
    off = np.asarray(ctx.grasp_offset)
    # Three spots along the bin's x axis (-5, 0, +5 cm) so successive parts land side by side, not on top of each other.
    spot_dx = (-0.05, 0.0, 0.05)[ctx.place_slot % 3]
    target = np.array([region.cx + spot_dx + off[0], region.cy + off[1], region.wall_top_z + ctx.hover_above_bin])
    # Up to the clearance height first (never down), across at that height, then down to the hover.
    err, crossing = travel(sim, ctx, target, "move_to_bin", timeout_s=ctx.move_timeout_s, tol_m=0.015)
    state = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    held = part_in_hand(state, ctx)
    data = {"err": err, "held": held, "crossing_z": crossing}
    if err > 0.03:
        return SkillResult(False, f"bin hover missed by {err:.3f} m", data)
    if not held:
        return SkillResult(False, "part dropped on the way to the bin", data)
    return SkillResult(True, "", data)


@skill("lower")
def lower(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """From the hover, bring the part straight down into the bin as far as the hand allows before letting go.

    The descent is guarded like a crossing: if the hand touches a tray or the table on the way down it stops there,
    the bump is reported, and the part is let go from that height (pushing on into the wall is what it must not do)."""
    region = bin_seen(ctx, ctx.target_bin)
    here = sim.wrist_pose(ctx.arm).pos
    target = np.array([here[0], here[1], region.wall_top_z + ctx.release_above_wall])
    err = sim.move_wrist(ctx.arm, target, grasp_quat(sim, ctx), timeout_s=ctx.move_timeout_s, tol_m=0.01, stop_on=fixed_boxes(ctx))
    bumped = sim.last_bump
    if bumped is not None:
        report_bump(sim, ctx, "lower", None)
    state = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    held = part_in_hand(state, ctx)
    data = {"err": err, "held": held, "wrist_above_wall": float(sim.wrist_pose(ctx.arm).pos[2]) - region.wall_top_z, "bumped": bumped}
    if err > 0.02 and bumped is None:
        return SkillResult(False, f"lowering into the bin stopped {err:.3f} m short", data)
    if not held:
        return SkillResult(False, "part slipped out while lowering into the bin", data)
    return SkillResult(True, "", data)


def let_go(sim: Simulator, arm: Arm) -> GripKeeper | None:
    """Open the hand at once. Every opening goes through here or open_hand: both take the grip keeper off first, or it
    would close the hand again on the falling force. Returns the keeper that was holding, if any."""
    keeper = stop_grip_keeper(sim, arm)
    sim.set_hand(arm, 0.0)
    return keeper


def open_hand(sim: Simulator, ctx: TaskContext) -> GripKeeper | None:
    """Let go gradually: ramp the hand open over release_open_s instead of snapping it open. Returns the grip keeper
    that held the part until now (its summary says whether the grip held through the carry), if there was one."""
    keeper = stop_grip_keeper(sim, ctx.arm)
    start = sim.hand_closure(ctx.arm)
    n = max(1, int(ctx.release_open_s / sim.dt / 5))
    for k in range(1, n + 1):
        sim.set_hand(ctx.arm, start * (1 - k / n))
        sim.step(5)
    sim.set_hand(ctx.arm, 0.0)
    return keeper


@skill("release")
def release(sim: Simulator, ctx: TaskContext) -> SkillResult:
    if ctx.grasp_mode == "weld":
        sim.detach(ctx.target_part)
    keeper = open_hand(sim, ctx)
    state = wait_until_resting(sim, ctx, max_s=4.0)
    ok = in_bin(state, ctx.target_part, ctx.target_bin)
    pos = state.objects[ctx.target_part].pos
    data = {"part_pos": pos.tolist()}
    if keeper is not None:
        data["grip_keeper"] = keeper.summary()  # how the grip held from the grasp to here
    return SkillResult(ok, "" if ok else "part is not resting in the target bin", data)


@skill("go_home")
def go_home(sim: Simulator, ctx: TaskContext) -> SkillResult:
    # Up and across to a spot clear of both trays first, so the joint move home does not sweep the bin walls.
    leave_bin(sim, ctx, "go_home")
    sim.set_joints({**{j: 0.0 for j in (f"{ctx.arm}_{i}" for i in range(1, 8))}, **TRAVEL_POSE})
    sim.set_head(0.0, HEAD_DOWN)  # keep watching the table; level would point the head camera at nothing
    sim.settle(2.0)
    return SkillResult(True)


@skill("retract")
def retract(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """After a release: lift clear of the bin walls, cross to a spot clear of both trays, and start the arm back,
    without waiting for it to arrive.

    go_home waits 2 s for the arm to settle at home, a third of a part's cycle. The next part is fed and
    inspected while the arm is still on its way back, so nothing has to wait for it.
    """
    leave_bin(sim, ctx, "retract")
    sim.set_joints({f"{ctx.arm}_{i}": 0.0 for i in range(1, 8)} | ARM_HOME[ctx.arm])
    sim.settle(ctx.retract_clear_s)  # long enough for the arm to leave the bin camera's view
    return SkillResult(True)


# How far the base origin backs off the table's near edge before the dance (Oct 21, dup: "it's dancing into the
# desk"). At the arms-out beat (every arm joint 0) the hands are 0.24 m ahead of the torso and the hand is 19 cm
# long (scripts/probe_joint_axes.py, zero pose), and the quarter turns sweep that reach round the base: 0.8 m clears
# it with room to spare. An estimate from the probe's numbers, not a measured clearance.
DANCE_CLEAR_M = 0.8


def back_away_from_table(sim: Simulator, ctx: TaskContext, clear_m: float = DANCE_CLEAR_M) -> float:
    """Arms up out of the way, then reverse straight back until the base origin is ``clear_m`` behind the table's near
    edge, holding the heading. Returns the base x reached."""
    sim.set_joints(TRAVEL_POSE)
    sim.settle(0.5)
    target_x = ctx.layout.table_x[0] - clear_m
    for _ in range(int(20.0 / sim.dt / 5)):
        pose = sim.base_pose()
        dx = float(pose.pos[0]) - target_x
        if dx <= 0.01:
            break
        yaw = yaw_of(pose.quat)
        sim.drive(-float(np.clip(dx * 1.0, 0.08, 0.4)), float(np.clip(-1.5 * yaw, -0.5, 0.5)))
        sim.step(5)
    sim.stop()
    sim.settle(0.5)
    return float(sim.base_pose().pos[0])


@skill("macarena")
def macarena(sim: Simulator, ctx: TaskContext) -> SkillResult:
    """The gated finale: back away from the table, then arms out, arms crossed, hands on hips, a hip wiggle on the
    base, repeat. Off unless the sort asks for it (SortContext.dance)."""
    back_away_from_table(sim, ctx)
    arms = lambda **kw: {f"{side}_{i}": kw.get(f"j{i}", 0.0) for side in ("left", "right") for i in range(1, 8)}
    beats = [
        {**arms(j4=0.0), "head_pitch": 0.0},  # arms straight out front
        {**arms(j4=0.0), "left_1": 0.6, "right_1": 0.6},  # palms up (shoulder roll)
        {**arms(j4=1.5)},  # hands to shoulders
        {**arms(j4=1.5), "head_yaw": 0.5},  # look right
        {**arms(j4=1.5), "head_yaw": -0.5},  # look left
        {**arms(j4=1.4), "left_1": -0.5, "right_1": -0.5, "head_yaw": 0.0},  # hands on hips
    ]
    for cycle in range(2):
        for pose in beats:
            sim.set_joints(pose)
            sim.settle(0.6)
        sim.drive(0.0, 1.2)  # the quarter turn
        sim.settle(1.3)
        sim.stop()
        sim.settle(0.3)
    sim.set_joints({**TRAVEL_POSE, "head_yaw": 0.0, "head_pitch": 0.0})
    sim.settle(1.0)
    return SkillResult(True)


def wait_until_resting(sim: Simulator, ctx: TaskContext, max_s: float = 4.0, min_s: float = 0.5) -> WorldState:
    """Step until the target part has stopped moving (position stable over 0.3 s) or ``max_s`` passes."""
    sim.settle(min_s)
    quiet = 0
    previous = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    state = previous
    for _ in range(int((max_s - min_s) / 0.1)):
        sim.settle(0.1)
        state = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
        quiet = quiet + 1 if still(previous, state, ctx.target_part) else 0
        previous = state
        if quiet >= 3:
            break
    return state


def final_state(sim: Simulator, ctx: TaskContext) -> WorldState:
    return wait_until_resting(sim, ctx)
