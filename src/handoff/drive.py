"""Driving the MH3's wheeled base between spots (Nov 4, for the capsule cells).

The base is a differential drive: it can turn in place and drive straight, not slide sideways. So a move to a pose is
three moves: turn to face the goal, drive to it holding that heading, turn to the final heading. Parking under a cell's
table is always the last stretch driven straight in along the cell's +x, so nothing is swung under the table top.

The base turns about its drive-wheel axle (``AXLE_AHEAD`` ahead of the origin), so every move steers that axle
point: a turn in place leaves it where it is, and the origin follows from it. Steering the origin instead swung it up
to 20 cm sideways on every turn (probe_drive_cells.py, Nov 4, first run).

Everything here reads only ``sim.base_pose()``: on a real robot that is wheel odometry plus the base camera's
localisation, not the simulator's truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim.base import AXLE_AHEAD, Simulator, yaw_of

MAX_LINEAR = 0.4  # m/s; the kit clamps at 0.6
MAX_ANGULAR = 0.8  # rad/s; the kit clamps at 2.0
POS_TOL = 0.02  # m
YAW_TOL = 0.03  # rad, about 1.7 degrees
APPROACH = 0.55  # m behind a park spot, along the cell's -x: clear of the table top before turning
STEP = 5  # physics steps per control update
# How far a held-heading stretch may turn to take out a sideways error: holding the heading exactly left the approach
# spot's sideways miss in the park (40 to 58 mm, probe_pick_leaker.py, Nov 4); 6 degrees swings the torso's front
# corner by under 2 cm.
HOLD_SLACK = 0.1


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class DriveResult:
    ok: bool
    err_pos: float
    err_yaw: float
    seconds: float


def _here(sim: Simulator) -> tuple[float, float, float]:
    pose = sim.base_pose()
    return float(pose.pos[0]), float(pose.pos[1]), yaw_of(pose.quat)


def _axle(sim: Simulator) -> tuple[float, float, float]:
    """The axle point and the heading: what the wheels actually move."""
    x, y, yaw = _here(sim)
    return x + AXLE_AHEAD * float(np.cos(yaw)), y + AXLE_AHEAD * float(np.sin(yaw)), yaw


def axle_of(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Where the axle is when the base origin is at (x, y) facing ``yaw``."""
    return x + AXLE_AHEAD * float(np.cos(yaw)), y + AXLE_AHEAD * float(np.sin(yaw))


def turn_to(sim: Simulator, yaw: float, timeout_s: float = 15.0) -> float:
    """Turn in place to a heading; returns the heading error left."""
    for _ in range(int(timeout_s / sim.dt / STEP)):
        err = wrap(yaw - _here(sim)[2])
        if abs(err) <= YAW_TOL:
            break
        sim.drive(0.0, float(np.clip(2.0 * err, -MAX_ANGULAR, MAX_ANGULAR)) if abs(err) > 0.1 else float(np.sign(err) * 0.15))
        sim.step(STEP)
    sim.stop()
    sim.step(STEP)
    return wrap(yaw - _here(sim)[2])


def straight_to(sim: Simulator, x: float, y: float, reverse: bool = False, hold_yaw: float | None = None, timeout_s: float = 30.0) -> float:
    """Drive the axle point in a straight line to (x, y), steering to keep it dead ahead (or dead behind when
    reversing), or with ``hold_yaw`` staying within ``HOLD_SLACK`` of that heading (under a table, where a turn would
    swing the torso).
    Stops when the point is reached or passed; returns the distance left."""
    for _ in range(int(timeout_s / sim.dt / STEP)):
        px, py, yaw = _axle(sim)
        dx, dy = x - px, y - py
        dist = float(np.hypot(dx, dy))
        heading = np.array([np.cos(yaw), np.sin(yaw)]) * (-1.0 if reverse else 1.0)
        along = float(heading @ np.array([dx, dy]))
        if dist <= POS_TOL or along <= 0.0:
            break
        bearing = np.arctan2(dy, dx) if not reverse else np.arctan2(-dy, -dx)
        if hold_yaw is not None:  # aim at the point, but never more than HOLD_SLACK off the held heading
            aim = hold_yaw + float(np.clip(wrap(float(bearing) - hold_yaw), -HOLD_SLACK, HOLD_SLACK)) if dist > 0.05 else hold_yaw
            steer = wrap(aim - yaw)
        else:
            steer = wrap(float(bearing) - yaw) if dist > 0.05 else 0.0
        speed = float(np.clip(1.2 * along, 0.06, MAX_LINEAR))
        sim.drive(-speed if reverse else speed, float(np.clip(2.0 * steer, -0.5, 0.5)))
        sim.step(STEP)
    sim.stop()
    sim.step(STEP)
    px, py, _ = _axle(sim)
    return float(np.hypot(x - px, y - py))


def drive_to(sim: Simulator, x: float, y: float, yaw: float) -> DriveResult:
    """Turn, drive, turn: the base origin to (x, y) facing ``yaw`` (radians, world frame)."""
    t0 = sim.time
    gx, gy = axle_of(x, y, yaw)
    px, py, _ = _axle(sim)
    if np.hypot(gx - px, gy - py) > POS_TOL:
        turn_to(sim, float(np.arctan2(gy - py, gx - px)))
        straight_to(sim, gx, gy)
    turn_to(sim, yaw)
    px, py, now = _here(sim)
    err_pos, err_yaw = float(np.hypot(x - px, y - py)), wrap(yaw - now)
    return DriveResult(err_pos <= 2 * POS_TOL and abs(err_yaw) <= 2 * YAW_TOL, err_pos, err_yaw, sim.time - t0)


def tuck_in(sim: Simulator, x: float, y: float, yaw: float) -> DriveResult:
    """Park at (x, y, yaw) by way of the spot ``APPROACH`` behind it, ending with a straight drive in along ``yaw``."""
    t0 = sim.time
    ax, ay = x - APPROACH * np.cos(yaw), y - APPROACH * np.sin(yaw)
    drive_to(sim, float(ax), float(ay), yaw)
    straight_to(sim, *axle_of(x, y, yaw), hold_yaw=yaw)
    turn_to(sim, yaw)
    px, py, now = _here(sim)
    err_pos, err_yaw = float(np.hypot(x - px, y - py)), wrap(yaw - now)
    return DriveResult(err_pos <= 2 * POS_TOL and abs(err_yaw) <= 2 * YAW_TOL, err_pos, err_yaw, sim.time - t0)


def back_out(sim: Simulator) -> DriveResult:
    """Reverse straight back ``APPROACH`` along the current heading: out from under a table before any turn."""
    t0 = sim.time
    px, py, yaw = _axle(sim)
    gx, gy = px - APPROACH * np.cos(yaw), py - APPROACH * np.sin(yaw)
    left = straight_to(sim, float(gx), float(gy), reverse=True, hold_yaw=yaw)
    _, _, now = _here(sim)
    return DriveResult(left <= 2 * POS_TOL, left, wrap(yaw - now), sim.time - t0)


def shift_under(sim: Simulator, x: float, y: float, yaw: float) -> DriveResult:
    """From one spot under a table to another: straight back out, across, and straight back in."""
    px, py, _ = _here(sim)
    if np.hypot(x - px, y - py) <= POS_TOL:
        return DriveResult(True, float(np.hypot(x - px, y - py)), wrap(yaw - _here(sim)[2]), 0.0)
    back_out(sim)
    return tuck_in(sim, x, y, yaw)
