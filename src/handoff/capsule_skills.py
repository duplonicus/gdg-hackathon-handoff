"""The robot's moves at a capsule cell: park at the station once, pick a leaker, drop it in the bin on its side.

The pick is a pinch (Nov 5; dup: "attach them to the fingers and make the hand grasp", and at the capsule plant people
picked leakers with their fingers in thin nylon gloves): the hand comes palm down with the index finger and thumb open
around the capsule's ends, closes them onto it, and the capsule is welded to the index fingertip (the weld stands in
for friction the contact solver cannot hold on an 11 mm pill). The pinch closes along the capsule's length, so the
tray leaves room past each end for a fingertip (capsule_cell.CellLayout.pitch). Measured by probe_pinch.py --sweep.

The station (Nov 5, dup: "we need to cut all of the driving"): the robot tucks in once at the cell origin and only its
arms move. Each hand works its own half of the tray (the left hand where y >= 0) and drops into the leaker bin on its
side. Where the hand straight ahead falls short, it turns its fingers toward the robot's middle (HEADINGS_DEG): the
working point sits 12 cm past the wrist, so turning the hand moves it inward and nearer (probe_station.py, Nov 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capsule_cell import Cell, park_for
from .drive import shift_under, tuck_in
from .sim.base import Arm, Simulator, quat_from_yaw, quat_mul, yaw_of

ARM = "left"  # the Nov 4 one-hand moves (park_at, the probes that use it)
HAND_POINT = np.array([0.12, -0.045])  # the sideways hand's working point (Nov 4 weld; probe_station.py's walk mode)
HAND_BELOW_WRIST = 0.03  # m the sideways hand's underside hangs below the wrist (probe_station.py's walk mode)
# The palm-down pinch (probe_pinch.py --roll-deg 90 --sweep, Nov 5): the hand rolled 90 degrees about its pointing axis
# (left hand +90, right -90); the midpoint of the index and thumb tips, from the wrist, in the hand's heading (x ahead,
# y left, z up), closed on a capsule. Mirrored in y for the right hand.
GRASP_ROLL_DEG = {"left": 90.0, "right": -90.0}
PINCH = np.array([0.126, -0.040, -0.052])
FINGERS_OPEN = {"index": 0.8, "thumb_1": 1.2, "thumb_2": 0.45}  # tips 50 mm apart along the capsule
FINGERS_PINCH = {"index": 1.1, "thumb_1": 1.2, "thumb_2": 0.45}  # tips 39 mm apart (centres): on a 24 mm capsule's ends
HOVER = 0.0  # m the pinch point sits above the capsule's centre when the fingers close
LIFT = 0.10  # m up after the weld, before anything moves sideways
DROP_ABOVE_RIM = 0.01  # m above the rim: where the sideways hand let go (Nov 4; skills of the sort task's bins use their own)
# The pinch goes this far below the leaker bin's rim, inside the bin, before it lets go (Nov 6): let go 2 cm above the
# rim, 20 of sel02's 299 leakers ended outside the bins, 18 just short of the bin's inner wall and 2 fallen through the
# world (VALIDATION, "sel02's lost leakers"). The bin is 8 cm deep; the straight fingers stay above the rim.
DROP_INTO_BIN = 0.03
PARK_SLACK = 0.07  # m: how far off a park may land and still be called parked
# Hand headings tried in order, degrees toward the robot's middle (probe_station.py, Nov 5: of 86 sampled spots reached
# from the station, 62 straight ahead, 21 at 30 and 3 at 60).
HEADINGS_DEG = (0.0, 30.0, 60.0, 90.0)
CLEAR_Z = 1.00  # m: a wrist below this (the bin rim plus a lift is 0.96) is raised to it before the arms fold
LIFT_TOL = 0.02
# The pinch closes along the capsule only with the hand straight ahead: turned, the open tips (50 mm apart) land on the
# next column (25 mm pitch) and the closed ones miss the capsule's ends (review of fdfd388). The bin takes any heading.
PICK_HEADINGS = (0.0,)  # m: a lift-height miss bigger than this means that heading does not reach; try the next


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""
    err: float = 0.0


def side_of(local_y: float) -> Arm:
    """Which hand works a point: the left hand the tray's left half (y >= 0 in the cell frame), the right the rest."""
    return "left" if local_y >= 0.0 else "right"


def _home_quat(arm: Arm = ARM) -> np.ndarray:
    from .skills import HOME_QUAT

    return np.asarray(HOME_QUAT[arm], dtype=float)


def _turn(arm: Arm, heading_deg: float) -> float:
    """The hand's turn in radians for a heading toward the middle: clockwise for the left hand, anticlockwise for the right."""
    return float(np.radians(heading_deg)) * (-1.0 if arm == "left" else 1.0)


def hand_quat(base_yaw: float, arm: Arm = ARM, heading_deg: float = 0.0) -> np.ndarray:
    """Palm down, turned with the base and then by the hand's heading toward the middle."""
    r = np.radians(GRASP_ROLL_DEG[arm])
    roll = np.array([np.cos(r / 2.0), np.sin(r / 2.0), 0.0, 0.0])  # about the pointing axis (world x at home)
    return quat_mul(quat_from_yaw(base_yaw + _turn(arm, heading_deg)), quat_mul(roll, _home_quat(arm)))


# Viewer and film runs only: the robot's torso shows at the top edge of the whole-tray picture, so a turned head could
# change it; the selection log (sel02) and test05 keep the head level, as frozen (docs/FREEZE.md).
LOOK_AT_WORK = False


def _look(sim: Simulator, xy, z: float) -> None:
    """Turn the head to what the hands are working on. The pictures the models judge come from the fixed scene cameras;
    this is so the robot visibly watches its work (dup, Nov 6: "its head isn't looking down at the table")."""
    if not LOOK_AT_WORK:
        return
    from .skills import look_at

    look_at(sim, (float(xy[0]), float(xy[1]), float(z)))


def set_fingers(sim: Simulator, arm: Arm, pose: dict[str, float], settle: float = 0.4) -> None:
    sim.set_joints({f"{arm}_{k}": v for k, v in pose.items()})
    sim.settle(settle)


def wrist_for(point_xy: tuple[float, float], z: float, base_yaw: float, clearance: float, arm: Arm = ARM, heading_deg: float = 0.0) -> np.ndarray:
    """The wrist position that puts ``arm``'s pinch point over ``point_xy`` at height ``z + clearance``, the hand palm
    down and turned ``heading_deg`` toward the middle."""
    yaw = base_yaw + _turn(arm, heading_deg)
    px, py, pz = PINCH[0], PINCH[1] * (1.0 if arm == "left" else -1.0), PINCH[2]
    c, s = np.cos(yaw), np.sin(yaw)
    off = np.array([c * px - s * py, s * px + c * py])
    return np.array([point_xy[0] - off[0], point_xy[1] - off[1], z + clearance - pz])


def travel_pose(sim: Simulator) -> None:
    """Forearms up, every other arm joint back to zero: the sort task's TRAVEL_POSE sets only the elbows, and an arm
    left twisted by a drop into its bin could not find the far row from there (Nov 5 station shift, seed 1000: the
    right hand missed spot 69 twice after two drops; from this pose it reaches it within 3 mm)."""
    from .skills import TRAVEL_POSE

    for arm in ("left", "right"):  # a hand low over the tray goes straight up first: folding from there swept it
        pose = sim.wrist_pose(arm)  # through the capsules (review of 0ebefd4; the Nov 5 fresh reach map shoved the robot)
        if float(pose.pos[2]) < CLEAR_Z:
            sim.move_wrist(arm, np.array([pose.pos[0], pose.pos[1], CLEAR_Z]), pose.quat, timeout_s=1.5, tol_m=0.02)
    sim.set_joints({**{f"{arm}_{j}": 0.0 for arm in ("left", "right") for j in range(1, 8)}, **TRAVEL_POSE})
    sim.settle(0.6)


def park_station(sim: Simulator, cell: Cell) -> MoveResult:
    """Drive in under the table to the cell's station, the one spot both hands work the whole tray from."""
    travel_pose(sim)
    wx, wy = cell.to_world(0.0, 0.0)
    result = tuck_in(sim, wx, wy, float(np.radians(cell.yaw_deg)))
    ok = result.err_pos <= PARK_SLACK
    if ok:
        sim.brake(True, f"{cell.name}_table")  # parked: wheels locked until the shift backs out
        _look(sim, cell.to_world(*cell.layout.tray_center), cell.layout.paper_top)
    return MoveResult(ok, "" if ok else f"parked {result.err_pos * 1000:.0f} mm off", result.err_pos)


def base_drift(sim: Simulator, home: tuple[float, float, float]) -> tuple[float, float]:
    """How far the base has moved from ``home`` (x, y, yaw): (metres, degrees). A sensor for the arms shoving the robot;
    the turn is read too, since a push far out on an arm turns the base more than it moves its origin."""
    pose = sim.base_pose()
    turn = (yaw_of(pose.quat) - home[2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.hypot(float(pose.pos[0]) - home[0], float(pose.pos[1]) - home[1])), float(np.degrees(abs(turn)))


def park_at(sim: Simulator, cell: Cell, local_xy: tuple[float, float], first: bool = False) -> MoveResult:
    """The Nov 4 one-hand move: drive so the left hand can work ``local_xy`` (cell frame) on the table."""
    bx, by = park_for(cell.layout, *local_xy)
    wx, wy = cell.to_world(bx, by)
    travel_pose(sim)
    result = (tuck_in if first else shift_under)(sim, wx, wy, float(np.radians(cell.yaw_deg)))
    # The hand's band has room to spare (park_for aims at its middle), so a few centimetres off still reaches; whether
    # it did is the next move's error. Only a miss past PARK_SLACK is called a failed park.
    ok = result.err_pos <= PARK_SLACK
    return MoveResult(ok, "" if ok else f"parked {result.err_pos * 1000:.0f} mm off", result.err_pos)


def _settle_on(sim: Simulator, arm: Arm, target: np.ndarray, quat: np.ndarray, tries: int = 4, gain: float = 0.005) -> float:
    """Move the wrist to ``target``, again while each try still closes in by more than ``gain``. From the forearms-up
    pose one 3 s move does not get there: spot 258 in cell a read 51, 42 and 31 mm on three tries in a row at
    heading 0, and 66, 39, 6 at 30 (Nov 5)."""
    err = float(sim.move_wrist(arm, target, quat, timeout_s=3.0, tol_m=0.01))
    for _ in range(tries - 1):
        if err <= 0.01:
            break
        again = float(sim.move_wrist(arm, target, quat, timeout_s=3.0, tol_m=0.01))
        closed, err = err - again, again
        if closed <= gain:
            break
    return err


def reach_over(sim: Simulator, arm: Arm, point_xy, top_z: float, clearance: float, tol: float, headings: tuple[float, ...] = HEADINGS_DEG) -> tuple[float, float, np.ndarray]:
    """Bring ``arm``'s hand to LIFT over ``point_xy``, then down to ``clearance``, trying the headings in turn until one
    reaches. Returns (error at ``clearance``, the heading used, its hand quaternion); a heading that misses at LIFT is
    not taken lower, so the hand never gropes down near the capsules on a heading that cannot get there."""
    yaw = yaw_of(sim.base_pose().quat)
    best = (float("inf"), headings[0], hand_quat(yaw, arm))
    down_on = None  # the heading whose descent the hand is resting at, if it is resting at one
    for k, heading in enumerate(headings):
        quat = hand_quat(yaw, arm, heading)
        above = wrist_for(point_xy, top_z, yaw, LIFT, arm, heading)
        down_on = None
        if _settle_on(sim, arm, above, quat) > LIFT_TOL:
            continue
        err = float(sim.move_wrist(arm, wrist_for(point_xy, top_z, yaw, clearance, arm, heading), quat, timeout_s=3.0, tol_m=tol))
        if err <= tol:
            return err, heading, quat
        best = min(best, (err, heading, quat), key=lambda b: b[0])
        down_on = heading
        if k < len(headings) - 1:  # back up only to turn the hand: with no heading left, stay down
            sim.move_wrist(arm, above, quat, timeout_s=2.0, tol_m=0.02)
            down_on = None
    # No heading within tol: end where the best one got to, so the hand is where the returned error was measured
    # (review of 72a8c9f: it ended at LIFT on the last heading, and a near-miss pick welded from 10 cm up). If the hand
    # is already down on the best heading, it stays: going up and down again on the same heading was the "down, up,
    # down, grab" dup saw on every pick (Nov 6), with the pick's single heading.
    err, heading, quat = best
    if np.isfinite(err) and down_on != heading:
        sim.move_wrist(arm, wrist_for(point_xy, top_z, yaw, LIFT, arm, heading), quat, timeout_s=3.0, tol_m=0.01)
        err = float(sim.move_wrist(arm, wrist_for(point_xy, top_z, yaw, clearance, arm, heading), quat, timeout_s=3.0, tol_m=tol))
    return err, heading, quat


def pick_capsule(sim: Simulator, cell: Cell, name: str, top_z: float, arm: Arm = ARM) -> MoveResult:
    """From where the robot stands: fingers open, the pinch point down onto the capsule's centre, close the index finger
    and thumb on its ends, weld it to the index fingertip, lift. (``top_z`` is kept for callers; the capsule's own
    centre is where the pinch goes.)"""
    at = sim.object_state(name).pos
    _look(sim, (at[0], at[1]), float(at[2]))
    # The open pinch is already a curl (index 0.8 of 1.1): set it with no wait so it forms during the reach down. Waiting
    # for it in the air looked like a first grab before the real one (dup, watching, Nov 6: "why is it grabbing twice").
    set_fingers(sim, arm, FINGERS_OPEN, settle=0.0)
    err, heading, quat = reach_over(sim, arm, (at[0], at[1]), float(at[2]), HOVER, 0.005, PICK_HEADINGS)
    if err > 0.015:
        return MoveResult(False, f"hand missed the capsule by {err * 1000:.0f} mm", err)
    set_fingers(sim, arm, FINGERS_PINCH, settle=0.5)
    sim.attach(name, arm, f"{arm}_index_tip")
    yaw = yaw_of(sim.base_pose().quat)
    sim.move_wrist(arm, wrist_for((at[0], at[1]), float(at[2]), yaw, LIFT, arm, heading), quat, timeout_s=3.0, tol_m=0.01)
    lifted = float(sim.object_state(name).pos[2]) - float(at[2])
    if lifted < LIFT * 0.5:
        sim.detach(name)
        set_fingers(sim, arm, FINGERS_OPEN)
        return MoveResult(False, f"capsule rose only {lifted * 1000:.0f} mm", err)
    return MoveResult(True, "", err)


def drop_in_bin(sim: Simulator, cell: Cell, name: str, arm: Arm = ARM) -> MoveResult:
    """From where the robot stands: straight up to the carry height, across over the middle of ``arm``'s leaker bin,
    down inside it below the rim, open the fingers, let go. Straight up first: from the pick's lift (about rim height) a straight
    line to the bin clipped its wall on every carry (dup, watching the viewer, Nov 5)."""
    lay = cell.layout
    middle = cell.to_world(*lay.leaker_bin(arm))
    rim = lay.table_top + lay.leaker_bin_floor + lay.leaker_bin_wall
    _look(sim, middle, lay.table_top)
    here = sim.wrist_pose(arm)
    carry_z = float(wrist_for(middle, rim, yaw_of(sim.base_pose().quat), LIFT, arm)[2])
    if float(here.pos[2]) < carry_z - 0.005:
        sim.move_wrist(arm, np.array([here.pos[0], here.pos[1], carry_z]), here.quat, timeout_s=3.0, tol_m=0.01)
    err, heading, quat = reach_over(sim, arm, middle, rim, -DROP_INTO_BIN, 0.01)
    # Open the fingers while the capsule is still welded to the index tip, then let go: released first, the opening
    # fingers could flick the free capsule off the bin's near wall.
    set_fingers(sim, arm, FINGERS_OPEN, settle=0.4)
    sim.detach(name)
    sim.settle(1.0)
    yaw = yaw_of(sim.base_pose().quat)
    sim.move_wrist(arm, wrist_for(middle, rim, yaw, LIFT, arm, heading), quat, timeout_s=3.0, tol_m=0.02)
    return MoveResult(err <= 0.02, "" if err <= 0.02 else f"hand missed the bin by {err * 1000:.0f} mm", err)


def remove_leaker(sim: Simulator, cell: Cell, name: str) -> MoveResult:
    """The whole move from the station: the hand on the capsule's side picks it and drops it in that side's bin."""
    lay = cell.layout
    pos = sim.object_state(name).pos
    arm = side_of(_to_local(cell, float(pos[0]), float(pos[1]))[1])
    result = pick_capsule(sim, cell, name, lay.paper_top + lay.capsule[2], arm)
    if not result.ok:
        sim.detach(name)  # never carry a capsule into the next move
        return result
    return drop_in_bin(sim, cell, name, arm)


def _to_local(cell: Cell, x: float, y: float) -> tuple[float, float]:
    c, s = np.cos(np.radians(cell.yaw_deg)), np.sin(np.radians(cell.yaw_deg))
    dx, dy = x - cell.origin[0], y - cell.origin[1]
    return (float(c * dx + s * dy), float(-s * dx + c * dy))
