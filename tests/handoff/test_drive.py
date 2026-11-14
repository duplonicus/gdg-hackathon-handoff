"""Driving the base on the fake simulator's unicycle model (Nov 4)."""

from __future__ import annotations

import numpy as np
import pytest

from handoff.drive import APPROACH, back_out, drive_to, shift_under, tuck_in, wrap
from handoff.sim.base import SceneSpec, yaw_of
from handoff.sim.fake import FakeSimulator

UNDER = -0.98  # base x past which its front (0.34 m ahead) is under cell a's table top (near edge at x -0.64)
TABLE_EDGE = -0.64  # cell a's near table edge
TORSO_HALF = (0.14, 0.15)  # the lift column and torso, around the base origin (probe_body.py): the part that cannot go under


def torso_front(p) -> float:
    """The furthest world x the torso reaches at pose p (its box turned with the base)."""
    return p[0] + abs(np.cos(p[2])) * TORSO_HALF[0] + abs(np.sin(p[2])) * TORSO_HALF[1]


def fresh(x=-1.2, y=0.0, yaw_deg=0.0) -> FakeSimulator:
    sim = FakeSimulator()
    sim.build(SceneSpec(robot_start=(x, y, yaw_deg)))
    return sim


def pose(sim):
    p = sim.base_pose()
    return float(p.pos[0]), float(p.pos[1]), yaw_of(p.quat)


@pytest.mark.parametrize("goal", [(-0.8, 0.0, 0.0), (-1.6, 0.0, np.pi), (-1.0, 0.4, np.pi / 2), (-1.6, -0.3, -np.pi)])
def test_drive_to_reaches_the_pose(goal):
    sim = fresh()
    result = drive_to(sim, *goal)
    x, y, yaw = pose(sim)
    assert result.ok
    assert np.hypot(x - goal[0], y - goal[1]) <= 0.04
    assert abs(wrap(yaw - goal[2])) <= 0.06


def test_tuck_in_ends_driving_straight_in_along_the_heading():
    """The last stretch into a table is straight: the base passes the approach spot facing the park heading."""
    sim = fresh()
    seen = []
    sim.add_controller("watch", lambda: seen.append(pose(sim)))
    result = tuck_in(sim, -0.8, 0.2, 0.0)
    assert result.ok
    assert max(torso_front(p) for p in seen) <= TABLE_EDGE  # the torso never reaches the table, however it turns
    under = [p for p in seen if p[0] > UNDER and abs(wrap(p[2])) < np.pi / 2]  # the base's front under the table
    assert under, "never drove the last stretch"
    assert max(abs(wrap(p[2])) for p in under) <= 0.1
    assert max(abs(p[1] - 0.2) for p in under) <= 0.03


def test_shift_under_backs_out_before_moving_sideways():
    sim = fresh(-0.8, 0.0, 0.0)
    seen = []
    sim.add_controller("watch", lambda: seen.append(pose(sim)))
    result = shift_under(sim, -0.8, -0.25, 0.0)
    assert result.ok
    under = [p for p in seen if p[0] > UNDER]
    assert under
    assert max(abs(wrap(p[2])) for p in under) <= 0.1  # never turned under the table
    assert all(abs(p[1]) <= 0.03 or abs(p[1] + 0.25) <= 0.03 for p in under)  # only at the old or the new y: moved across once out


def test_back_out_reverses_along_the_heading():
    sim = fresh(-0.8, 0.1, 0.0)
    result = back_out(sim)
    x, y, yaw = pose(sim)
    assert result.ok
    assert x == pytest.approx(-0.8 - APPROACH, abs=0.04) and y == pytest.approx(0.1, abs=0.02) and abs(yaw) <= 0.03
