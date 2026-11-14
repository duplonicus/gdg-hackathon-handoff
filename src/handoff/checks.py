"""Success checks: plain functions of a world-state snapshot.

A check never looks at the simulator directly. It reads a :class:`WorldState` taken with
:func:`snapshot` at any step, so the same functions decide a trial, label logged decisions with
what really happened, and serve as rewards for a learned grasp later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .scene import BinRegion, Layout
from .sim.base import Arm, ObjectState, Pose, Simulator


@dataclass(frozen=True)
class WorldState:
    t: float
    objects: Mapping[str, ObjectState]
    wrists: Mapping[str, Pose]
    hand_closure: Mapping[str, float]
    contacts: Mapping[tuple[str, str], int]  # (arm, object) -> contact points
    table_top: float
    bins: Mapping[str, BinRegion]
    part_size: tuple[float, float, float]


def snapshot(sim: Simulator, layout: Layout, arms: tuple[Arm, ...] = ("left",), parts: tuple[str, ...] | None = None) -> WorldState:
    """Snapshot the parts in play (default: the whole pool, which includes the parked ones)."""
    parts = layout.part_names if parts is None else parts
    return WorldState(
        t=sim.time,
        objects={name: sim.object_state(name) for name in parts},
        wrists={arm: sim.wrist_pose(arm) for arm in arms},
        hand_closure={arm: sim.hand_closure(arm) for arm in arms},
        contacts={(arm, name): sim.hand_contacts(arm, name) for arm in arms for name in parts},
        table_top=layout.table_top,
        bins=layout.bin_regions(),
        part_size=layout.part_size,
    )


def speed(state: WorldState, part: str) -> float:
    return float(np.linalg.norm(state.objects[part].lin_vel))


REST_SPEED = 0.05  # m/s; only used for "is it falling", see below


def settled(state: WorldState, part: str, max_speed: float = REST_SPEED) -> bool:
    """Not moving by its reported velocity. Unreliable for a mesh part resting against a tray wall:
    the contact solver reports a steady 0.06 to 0.45 m/s on a part that never moves (seen Oct 11).
    Use :func:`still` (position over time) to decide rest; keep this for "is it falling"."""
    return speed(state, part) <= max_speed


def still(before: WorldState, after: WorldState, part: str, tol: float = 0.005) -> bool:
    """The part has not moved between two snapshots (the reliable test of rest)."""
    return float(np.linalg.norm(after.objects[part].pos - before.objects[part].pos)) <= tol


def hand_touching(state: WorldState, part: str, arm: Arm = "left") -> bool:
    return state.contacts.get((arm, part), 0) > 0


def part_near_wrist(state: WorldState, part: str, arm: Arm = "left", max_dist: float = 0.20) -> bool:
    d = float(np.linalg.norm(state.objects[part].pos - state.wrists[arm].pos))
    return d <= max_dist


def lifted(state: WorldState, part: str, min_lift: float = 0.03) -> bool:
    """Part bottom is above the table top by ``min_lift`` (it is off the table, not just tilted)."""
    bottom = float(state.objects[part].pos[2]) - state.part_size[2] / 2.0
    return bottom >= state.table_top + min_lift


def grasp_held(state: WorldState, part: str, arm: Arm = "left") -> bool:
    """The hand is holding the part: it is close to the wrist, touching the hand, and off the table."""
    return part_near_wrist(state, part, arm) and hand_touching(state, part, arm) and lifted(state, part)


def moved_with_hand(before: WorldState, after: WorldState, part: str, arm: Arm = "left", tol: float = 0.03) -> bool:
    """Over a move, the part travelled with the wrist (displacements agree within ``tol``)."""
    d_part = after.objects[part].pos - before.objects[part].pos
    d_wrist = after.wrists[arm].pos - before.wrists[arm].pos
    return float(np.linalg.norm(d_part - d_wrist)) <= tol


def in_bin(state: WorldState, part: str, bin_name: str, max_speed: float | None = None) -> bool:
    """Part centre inside the bin's inner footprint and within the tray's height band.

    Geometry only: the reported velocity of a part lying in a tray is not trustworthy. Callers
    that need the part to have come to rest wait with :func:`still` first.
    """
    region = state.bins[bin_name]
    pos = state.objects[part].pos
    half = region.half_inner
    inside_xy = abs(float(pos[0]) - region.cx) <= half and abs(float(pos[1]) - region.cy) <= half
    bottom = float(pos[2]) - min(state.part_size) / 2.0  # the part may lie on any face
    # From the tray floor up to one part stacked on another; a part balanced on a wall is outside the footprint anyway.
    in_band = region.floor_z - 0.01 <= bottom <= region.wall_top_z + max(state.part_size)
    falling = max_speed is not None and speed(state, part) > max_speed
    return inside_xy and in_band and not falling


def on_floor(state: WorldState, part: str) -> bool:
    """The part fell off the table."""
    return float(state.objects[part].pos[2]) < state.table_top - 0.05


def any_on_floor(state: WorldState) -> list[str]:
    return [name for name in state.objects if on_floor(state, name)]
