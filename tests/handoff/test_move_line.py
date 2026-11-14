"""GenesisSimulator.move_wrist follows the straight line to its target, not the arc the joints make between two poses.

Oct 21: interpolating the joints straight from start to goal sagged a 45 cm crossing from 1.03 to 0.945 m, over both
trays' wall tops, and the bump guard stopped every carry (scripts/probe_crossing_trace.py). The kit here is a stub: a
two-link planar arm on left_1 and left_2 (the other joints stay at 0) with exact forward and inverse kinematics, and
joints that go exactly where they are commanded.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

L1, L2 = 0.4, 0.4
JOINTS = tuple(f"left_{i}" for i in range(1, 8))


def fk(q: np.ndarray) -> np.ndarray:
    """Wrist position of the planar arm: x forward, z up, y 0."""
    a, b = float(q[0]), float(q[1])
    return np.array([L1 * np.cos(a) + L2 * np.cos(a + b), 0.0, L1 * np.sin(a) + L2 * np.sin(a + b)])


def ik(pos: np.ndarray) -> np.ndarray:
    """Elbow-up solution for a wrist position in the x-z plane (the other five joints 0)."""
    x, z = float(pos[0]), float(pos[2])
    c2 = (x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    b = -np.arccos(np.clip(c2, -1.0, 1.0))
    a = np.arctan2(z, x) - np.arctan2(L2 * np.sin(b), L1 + L2 * np.cos(b))
    return np.array([a, b, 0.0, 0.0, 0.0, 0.0, 0.0])


class PlanarKit:
    _public_q_indices: ClassVar[dict[str, int]] = {name: i for i, name in enumerate(JOINTS)}

    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=float)
        self.commanded: list[np.ndarray] = []

    def get_link_pose(self, name, frame=None):
        return fk(self.q), np.array([1.0, 0.0, 0.0, 0.0])

    def solve_ik(self, link, position, orientation=None, joints=None, apply=False, **options):
        return ik(np.asarray(position, dtype=float))

    def set_joint_positions(self, targets, immediate=False):
        self.q = np.array([targets.get(name, self.q[i]) for i, name in enumerate(JOINTS)])
        self.commanded.append(self.q.copy())

    def get_joint_state(self, names):
        class State:
            pass

        s = State()
        s.names, s.positions = list(names), [float(self.q[JOINTS.index(n)]) for n in names]
        return s

    def step(self):
        pass


def test_a_level_wrist_move_stays_on_the_line_the_whole_way():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    start, goal = np.array([0.30, 0.0, 0.20]), np.array([0.70, 0.0, 0.20])
    kit = PlanarKit(ik(start))
    sim = genesis_sim.GenesisSimulator()
    sim._sim = kit
    err = sim.move_wrist("left", goal, np.array([1.0, 0.0, 0.0, 0.0]), timeout_s=0.1, tol_m=0.005)
    assert err < 0.005
    heights = [fk(q)[2] for q in kit.commanded]
    assert len(heights) > 20
    # Joint interpolation between the two poses bows this 40 cm move off its height by centimetres; the line holds
    # it within the chord sag of a 1 cm step.
    assert max(abs(h - 0.20) for h in heights) < 0.002


def test_along_interpolates_between_equally_spaced_points():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    path = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0]])
    assert np.allclose(genesis_sim.along(path, 0.0), [0.0, 0.0])
    assert np.allclose(genesis_sim.along(path, 0.25), [0.5, 1.0])
    assert np.allclose(genesis_sim.along(path, 0.75), [2.0, 2.0])
    assert np.allclose(genesis_sim.along(path, 1.0), [3.0, 2.0])
    assert np.allclose(genesis_sim.along(path[:1], 0.5), [0.0, 0.0])


def test_slerp_takes_the_short_way_and_stays_unit():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])  # the same rotation as +, sign flipped
    mid = genesis_sim.slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0)
    assert np.allclose(mid, [np.cos(0.25), 0.0, 0.0, np.sin(0.25)])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def genesis_masked_rot_error(hand: np.ndarray, target: np.ndarray, rot_mask: list[bool]) -> np.ndarray:
    """Genesis 1.4.3's IK rotation error as it masks it: rotvec(target * inv(hand)), a WORLD-frame vector
    (inverse_kinematics.py), with a one-True mask inverted first (rigid_entity.inverse_kinematics)."""
    mask = np.array(rot_mask, dtype=bool)
    if mask.sum() == 1:
        mask = ~mask
    inv = hand * np.array([1.0, -1.0, -1.0, -1.0])
    q = qmul(target, inv)
    q = q if q[0] >= 0 else -q
    angle = 2 * np.arctan2(np.linalg.norm(q[1:]), q[0])
    rotvec = q[1:] / max(np.linalg.norm(q[1:]), 1e-12) * angle
    return rotvec * mask


def test_the_free_yaw_mask_frees_world_yaw_and_holds_the_tilt_as_genesis_applies_it():
    """The Oct 21 review: the first mask named the hand's own up axis, which Genesis reads per WORLD axis, so it froze
    the yaw and freed the pitch."""
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    from handoff.sim.base import quat_from_yaw, quat_mul
    from handoff.skills import HOME_QUAT

    grasp = quat_mul(quat_from_yaw(np.radians(10.0)), HOME_QUAT["left"])
    yawed = quat_mul(quat_from_yaw(np.radians(15.0)), grasp)  # the hand turned about world vertical
    pitched = qmul(np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0]), grasp)  # tipped 11.5 degrees about world y
    mask = genesis_sim.YAW_FREE_MASK
    assert np.linalg.norm(genesis_masked_rot_error(yawed, grasp, mask)) < 1e-9  # a turn about vertical is free
    assert np.linalg.norm(genesis_masked_rot_error(pitched, grasp, mask)) > 0.19  # a tilt is held
    old = [False, True, False]  # 1d0a0e8: the hand's own up axis
    assert np.linalg.norm(genesis_masked_rot_error(yawed, grasp, old)) > 0.25  # froze the yaw
    assert np.linalg.norm(genesis_masked_rot_error(pitched, grasp, old)) < 1e-9  # and freed the tilt


def test_a_free_yaw_move_asks_the_ik_for_the_level_axis_only():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    kit = PlanarKit(ik(np.array([0.30, 0.0, 0.20])))
    asked = []
    real = kit.solve_ik

    def solve_ik(link, position, orientation=None, joints=None, apply=False, **options):
        asked.append(options)
        return real(link, position, orientation, joints, apply, **options)

    kit.solve_ik = solve_ik
    sim = genesis_sim.GenesisSimulator()
    sim._sim = kit
    sim.move_wrist("left", np.array([0.40, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0]), timeout_s=0.1, tol_m=0.005, free_yaw=True)
    assert asked and all(o.get("rot_mask") == genesis_sim.YAW_FREE_MASK for o in asked)
    assert "init_qpos" not in asked[0] and all("init_qpos" in o for o in asked[1:])  # each point seeded with the last
    asked.clear()
    sim.move_wrist("left", np.array([0.30, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0]), timeout_s=0.1, tol_m=0.005)
    assert asked and all("rot_mask" not in o for o in asked)


def flipping_kit(fresh_extra: float) -> PlanarKit:
    """A planar arm whose IK, seeded from a point past the middle of the line, restarts on another branch: joint 3 (not
    in the planar FK) +1 rad, and every later seeded point stays there, as Genesis's chained seeds do. Solved unseeded
    (from the current pose), the goal (x 0.70) comes back with ``fresh_extra`` on joint 3."""
    kit = PlanarKit(ik(np.array([0.30, 0.0, 0.20])))

    def solve_ik(link, position, orientation=None, joints=None, apply=False, **options):
        q = ik(np.asarray(position, dtype=float)).copy()
        if "init_qpos" not in options:  # the line's first point, or the goal re-solved after a flip
            q[2] += fresh_extra if position[0] > 0.65 else 0.0
        elif position[0] > 0.5 or options["init_qpos"][2] > 0.5:
            q[2] += 1.0
        return q

    kit.solve_ik = solve_ik
    return kit


def test_a_line_whose_ik_flips_branch_falls_back_to_the_joint_path_instead_of_whipping():
    """The Oct 21 review: one unconverged IK point on another arm branch would be played back inside a 1 cm segment.
    The Oct 22 nightly review: the fallback then eased to the flipped branch's answer for the goal; it goes to the goal
    solved fresh from the start pose, and the run log can show it."""
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    kit = flipping_kit(fresh_extra=0.0)
    sim = genesis_sim.GenesisSimulator()
    sim._sim = kit
    sim.move_wrist("left", np.array([0.70, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0]), timeout_s=0.1, tol_m=0.005)
    commanded = np.array(kit.commanded)
    assert np.abs(np.diff(commanded, axis=0)).max() < 0.1  # eased in joint space: no single-step whip
    assert abs(commanded[-1][2]) < 1e-9  # ends on the arm's own branch, not the flipped one (joint 3 +1 rad)
    assert np.allclose(commanded[-1], ik(np.array([0.70, 0.0, 0.20])))
    assert sim.line_fallbacks == 1
    (info,) = sim.line_fallback_log
    assert info["goal"] == "fresh" and info["jump_rad"] > genesis_sim.LINE_JUMP_RAD and info["arm"] == "left"


def test_a_flip_keeps_the_line_s_answer_when_the_fresh_goal_swings_further():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    kit = flipping_kit(fresh_extra=2.0)  # the unseeded solve lands even further from the start joints
    sim = genesis_sim.GenesisSimulator()
    sim._sim = kit
    sim.move_wrist("left", np.array([0.70, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0]), timeout_s=0.1, tol_m=0.005)
    assert np.isclose(kit.commanded[-1][2], 1.0)
    (info,) = sim.line_fallback_log
    assert info["goal"] == "chained" and info["swing_rad"] >= 1.0


def test_a_line_fallback_reads_as_one_run_log_line():
    from handoff.sim.base import line_fallback_line

    line = line_fallback_line({"t": 12.345, "arm": "left", "target": [0.7, 0.1, 1.03], "jump_rad": 1.0, "goal": "fresh", "swing_rad": 0.62})
    assert "t 12.35 s" in line and "left wrist to 0.70, 0.10, 1.03" in line and "1.00 rad" in line
    assert "solved fresh from the start pose" in line and "0.62 rad" in line
    assert "chained" in line_fallback_line({"t": 0, "arm": "right", "target": [0, 0, 0], "jump_rad": 1, "goal": "chained", "swing_rad": 1})
