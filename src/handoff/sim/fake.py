"""A kinematic fake of the simulator, for tests that must not need the GPU.

It teleports instead of simulating: a wrist move succeeds when the target is inside a reach
sphere around the shoulder, a closed hand holds a part that was within hand reach at closing,
and a free part drops straight down onto whatever is under it (bin floor, table, ground). It is
not physics; it exists so the skill sequence, the checks and the trial runner have a fast,
deterministic world to run against.

Its sensors read noise-free and deterministic, shaped by the same part specs as the real ones (an
IMU clips at its range, a fingertip reads 0 below its actuation force), and tests script them:
``inject_imu`` adds a spike to one placement for some steps, ``set_joint_torques``,
``set_tactile``, ``set_fingertips`` and ``set_finger_penetration`` set what the next reads return.
Unscripted, an IMU reads gravity (plus the base's turn rate) and each fingertip presses on the part
in the hand by how far the closure squeezes past the part's width.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..bump import record_stop, start_guard
from .base import (
    AXLE_AHEAD,
    G0,
    Arm,
    BoxSpec,
    Frame,
    ObjectState,
    Pose,
    SceneSpec,
    SensorConfig,
    Simulator,
    fingertip_links,
    quat_from_yaw,
    quat_mul,
    skin_links,
    yaw_of,
)

SHOULDER = {"left": np.array([0.0, 0.15, 1.20]), "right": np.array([0.0, -0.15, 1.20])}
HOME_WRIST = {"left": np.array([0.32, 0.22, 0.885]), "right": np.array([0.32, -0.22, 0.885])}
HOME_QUAT = np.array([0.5, 0.5, 0.5, 0.5])
REACH_M = 0.63
HAND_REACH_M = 0.16  # a part this close to the wrist can be held
HOLD_OFFSET = np.array([0.12, -0.045, 0.0])  # where a held part sits relative to the wrist (left hand)
# The fake hand's fingertip model (a stand-in, not a measured hand): the gap between fingertips and thumb shrinks
# linearly from OPEN_APERTURE_M at closure 0 to nothing at 1, and each fingertip presses with TIP_STIFFNESS_N_PER_M
# times however far that gap is commanded inside the part's narrower side.
OPEN_APERTURE_M = 0.10
TIP_STIFFNESS_N_PER_M = 200.0


@dataclass
class ImuSpike:
    """An injected IMU reading added on top of the normal one for the next ``steps`` physics steps."""

    placement: str
    acc: np.ndarray
    gyro: np.ndarray
    steps: int


@dataclass(frozen=True)
class Move:
    """One move_wrist call as the fake saw it, for tests that check the path the skills ask for."""

    arm: str
    start: np.ndarray
    target: np.ndarray
    stop_on: tuple[str, ...]
    bump: str | None  # the injected bump that stopped it, else None


class FakeSimulator(Simulator):
    def __init__(self, dt: float = 0.01, sensors: SensorConfig | None = None) -> None:
        self._dt = dt
        self.sensors = SensorConfig() if sensors is None else sensors
        self._t = 0.0
        self._scene: SceneSpec | None = None
        self._objects: dict[str, np.ndarray] = {}
        self._yaws: dict[str, float] = {}
        self._vel: dict[str, np.ndarray] = {}
        self._wrist: dict[str, Pose] = {}
        self._closure: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._held: dict[str, str] = {}  # object -> arm (finger hold)
        self._welded: dict[str, str] = {}  # object -> arm
        self._weld_offset: dict[str, np.ndarray] = {}  # object -> its offset from the wrist in the wrist's frame, at the weld
        self._base = np.zeros(3)  # x, y, yaw
        self._drive = (0.0, 0.0)
        self._joints: dict[str, float] = {}
        self._head = (0.0, 0.0)
        self.calls: list[str] = []
        self.moves: list[Move] = []  # every move_wrist call, in order
        self._bumps: list[str] = []  # injected bumps, used up in order by guarded moves (inject_bump)
        self._spikes: list[ImuSpike] = []
        self._torques: dict[str, np.ndarray] = {}
        self._tactile: dict[str, dict[str, np.ndarray]] = {}
        self._fingertips: dict[str, dict[str, float]] = {}
        self._penetration: dict[tuple[str, str], dict[str, float]] = {}
        self._touch: dict | None = None  # the hand pressing on a fixed box during a guarded move (inject_bump)

    # ----- world -----
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return self._t

    def build(self, scene: SceneSpec) -> None:
        self._scene = scene
        self.reset()

    def reset(self) -> None:
        assert self._scene is not None
        self.clear_controllers()
        self._t = 0.0
        self._objects = {b.name: np.array(b.pos, dtype=float) for b in self._scene.boxes}
        self._yaws = {b.name: b.yaw_deg for b in self._scene.boxes}
        self._vel = {b.name: np.zeros(3) for b in self._scene.boxes}
        x, y, yaw = self._scene.robot_start
        yaw = np.deg2rad(yaw)
        c, s_ = np.cos(yaw), np.sin(yaw)
        # HOME_WRIST is with the base at the origin facing +x; the wrists start where the base does.
        self._wrist = {
            arm: Pose(np.array([x + c * w[0] - s_ * w[1], y + s_ * w[0] + c * w[1], w[2]]), quat_mul(quat_from_yaw(yaw), HOME_QUAT))
            for arm, w in HOME_WRIST.items()
        }
        self._closure = {"left": 0.0, "right": 0.0}
        self._held.clear()
        self._welded.clear()
        self._weld_offset.clear()
        x, y, yaw = self._scene.robot_start
        self._base = np.array([x, y, np.deg2rad(yaw)])
        self._drive = (0.0, 0.0)
        self._joints = {}
        self._spikes = []
        self._torques = {}
        self._tactile = {}
        self._fingertips = {}
        self._penetration = {}
        self._touch = None

    def _support(self, name: str) -> tuple[str | None, float]:
        """The fixed box under the object's centre and its top, else (None, ground)."""
        assert self._scene is not None
        pos = self._objects[name]
        best: tuple[str | None, float] = (None, 0.0)
        for b in self._scene.boxes:
            if not b.fixed or not b.collision or b.name == name:
                continue
            # where the box is now: place_episode moves the trays per episode
            center = self._objects.get(b.name, np.array(b.pos, dtype=float))
            half = np.array(b.size) / 2.0
            if abs(pos[0] - center[0]) <= half[0] and abs(pos[1] - center[1]) <= half[1]:
                top = center[2] + half[2]
                if top <= pos[2] + 1e-6 and top > best[1]:
                    best = (b.name, top)
        return best

    def _support_z(self, name: str) -> float:
        """Top of whatever fixed box lies under the object's centre, else the ground."""
        return self._support(name)[1]

    def tray_load(self, boxes: tuple[str, ...]) -> float:
        """The weight of every free part resting on one of these boxes (no physics: resting means at its support height)."""
        assert self._scene is not None
        total = 0.0
        for name in self._scene.free_names:
            if name in self._held or name in self._welded:
                continue
            box = self._scene.box(name)
            support, top = self._support(name)
            if support in boxes and abs(self._objects[name][2] - (top + box.size[2] / 2.0)) < 1e-6:
                total += box.rho * float(np.prod(box.size)) * 9.81
        return total

    def step(self, n: int = 1) -> None:
        assert self._scene is not None
        for _ in range(n):
            self._t += self._dt
            self._run_controllers()
            if self._touch is not None:
                self._touch["steps"] += 1
            for spike in self._spikes:
                spike.steps -= 1
            self._spikes = [s for s in self._spikes if s.steps > 0]
            lin, ang = self._drive  # a differential drive: the axle point moves, the origin trails it
            x, y, yaw = self._base
            ax, ay = x + AXLE_AHEAD * np.cos(yaw), y + AXLE_AHEAD * np.sin(yaw)
            yaw += ang * self._dt
            ax, ay = ax + lin * np.cos(yaw) * self._dt, ay + lin * np.sin(yaw) * self._dt
            new = np.array([ax - AXLE_AHEAD * np.cos(yaw), ay - AXLE_AHEAD * np.sin(yaw), yaw])
            if lin or ang:  # the arms ride on the base: both wrists move with it, and so does anything they hold
                turn = yaw - self._base[2]
                c, s_ = np.cos(turn), np.sin(turn)
                for arm, pose in self._wrist.items():
                    rel = pose.pos[:2] - self._base[:2]
                    xy = new[:2] + np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
                    self._wrist[arm] = Pose(np.array([xy[0], xy[1], pose.pos[2]]), quat_mul(quat_from_yaw(turn), pose.quat))
            self._base = new
            for name in self._scene.free_names:
                carrier = self._welded.get(name) or self._held.get(name)
                if carrier:
                    self._objects[name] = self._carried_point(name, carrier)
                    self._vel[name] = np.zeros(3)
                    continue
                half_h = self._scene.box(name).size[2] / 2.0
                rest = self._support_z(name) + half_h
                before = self._objects[name][2]
                self._objects[name][2] = max(rest, before - 9.81 * self._dt * 0.5)
                self._vel[name] = np.array([0.0, 0.0, (self._objects[name][2] - before) / self._dt])

    # ----- objects -----
    def object_state(self, name: str) -> ObjectState:
        return ObjectState(self._objects[name].copy(), quat_from_yaw(np.deg2rad(self._yaws[name])), self._vel[name].copy())

    def set_object_pose(self, name: str, pos: tuple[float, float, float], yaw_deg: float = 0.0) -> None:
        self._objects[name] = np.array(pos, dtype=float)
        self._yaws[name] = yaw_deg
        self._vel[name] = np.zeros(3)

    # ----- arms and hands -----
    def wrist_pose(self, arm: Arm) -> Pose:
        return self._wrist[arm]

    def inject_bump(self, box: str, times: int = 1) -> None:
        """Make the next ``times`` moves that guard against ``box`` (``stop_on``) touch it halfway and press there.

        The touch is not reported directly: it shows in the robot's sensors (``_start_touch``), and the move stops only
        if the configured bump detector (``config/bump.yaml``, or ``bump_config``) catches it, naming the box it
        estimates from the drawing, as the real robot does."""
        self._bumps.extend([box] * times)

    # The fake's signature of a hand touching a fixed box: stand-ins of plausible size for the detector tests, not
    # measured (the probe, scripts/probe_bump_sensors.py, measures the real ones): an impact on the hand IMU against
    # the motion for two samples, a little of it at the body IMUs, an external torque building on every arm joint, and
    # a force on the palm skin, for as long as the hand presses.
    TOUCH_HAND_MPS2 = 25.0
    TOUCH_BODY_MPS2 = 1.0
    TOUCH_TORQUE_RAMP_NM = 0.5  # per step, up to TOUCH_TORQUE_MAX_NM
    TOUCH_TORQUE_MAX_NM = 5.0
    TOUCH_PALM_N = 2.0
    bump_config = None  # a handoff.bump.BumpConfig; None: config/bump.yaml, read on the first guarded move
    bump_guard = None  # the last guarded move's detectors
    bump_log = None  # every detected bump's report
    last_bump_info = None
    bins_seen = None  # the task's bins_seen, for the box estimate

    def _start_touch(self, arm: Arm, box: str, direction: np.ndarray) -> None:
        """The hand meets ``box``: from now until ``_touch`` is cleared the sensors read the touch signature."""
        assert self._scene is not None
        self._touch = {"arm": arm, "box": box, "steps": 0}  # which box is truth: never handed to the box estimate
        norm = float(np.linalg.norm(direction))
        against = -direction / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        for p, _link in self.sensors.placements:
            size = self.TOUCH_HAND_MPS2 if p == f"{arm}_hand" else self.TOUCH_BODY_MPS2
            self._spikes.append(ImuSpike(p, against * size, np.zeros(3), 3))  # 3: read after two steps, then gone

    def torque_model(self, arm: Arm) -> tuple[np.ndarray, np.ndarray] | None:
        """The fake's dynamics model for the momentum observer: no motion, no gravity, so (dp, bias) are zero and any
        torque read is an external load."""
        return (np.zeros(7), np.zeros(7)) if arm in self.sensors.arms("torque") else None

    def _carried_point(self, name: str, arm: Arm) -> np.ndarray:
        """A welded part keeps its pose relative to the wrist from the moment of the weld, as a weld does; a part held
        by the fingers sits at the hold point."""
        offset = self._weld_offset.get(name)
        if offset is None:
            return self._hold_point(arm)
        pose = self._wrist[arm]
        return pose.pos + _rotate(pose.quat, offset)

    def _hold_point(self, arm: Arm) -> np.ndarray:
        """Where a held part sits: HOLD_OFFSET is for the left hand in its home heading, turns with the hand, and is
        mirrored for the right hand (its working point is on the inboard side too)."""
        pose = self._wrist[arm]
        turn = yaw_of(pose.quat) - yaw_of(HOME_QUAT)
        c, s_ = np.cos(turn), np.sin(turn)
        ox, oy, oz = HOLD_OFFSET
        oy = oy if arm == "left" else -oy
        return pose.pos + np.array([c * ox - s_ * oy, s_ * ox + c * oy, oz])

    def _shoulder(self, arm: Arm) -> np.ndarray:
        """The shoulder in the world: SHOULDER is measured with the base at the origin facing +x, and moves with it."""
        x, y, yaw = self._base
        c, s = np.cos(yaw), np.sin(yaw)
        sx, sy, sz = SHOULDER[arm]
        return np.array([x + c * sx - s * sy, y + s * sx + c * sy, sz])

    def move_wrist(
        self,
        arm: Arm,
        pos: np.ndarray,
        quat: np.ndarray,
        *,
        timeout_s: float = 3.0,
        tol_m: float = 0.01,
        duration_s: float | None = None,
        stop_on: tuple[str, ...] = (),
        free_yaw: bool = False,
    ) -> float:
        self.calls.append(f"move_wrist:{arm}")
        pos = np.asarray(pos, dtype=float)
        start = self._wrist[arm].pos.copy()
        self.last_bump = None
        quat = np.asarray(quat, dtype=float)
        shoulder = self._shoulder(arm)
        dist = float(np.linalg.norm(pos - shoulder))
        # Out of reach, the arm stretches toward the target and stops at the reach limit.
        reached = shoulder + (pos - shoulder) / dist * REACH_M if dist > REACH_M else pos.copy()
        if not stop_on:
            self.moves.append(Move(arm, start, pos.copy(), (), None))
            self._wrist[arm] = Pose(reached, quat)
            self.settle(timeout_s if dist > REACH_M else 0.5)
            return float(np.linalg.norm(pos - reached))
        # A guarded move goes along the straight line over 0.5 s, its bump detectors (handoff.bump) fed every step.
        bump = next((box for box in self._bumps if box in stop_on), None)
        if bump is not None:
            self._bumps.remove(bump)
        guard = start_guard(self, arm, stop_on)
        assert guard is not None
        n = max(2, int(round(0.5 / self._dt)))
        stopped = False
        for k in range(1, n + 1):
            if bump is not None and k == n // 2:
                self._start_touch(arm, bump, reached - start)  # the hand meets the box halfway and presses there
            if self._touch is None:
                self._wrist[arm] = Pose(start + (reached - start) * k / n, quat)
            self.step(1)
            if guard.step():
                stopped = True
                break
        touch, self._touch = self._touch, None  # the arm stops pushing, or the move ends pressed where it met the box
        if stopped:
            # Where the robot thinks its hand is (its kinematics: the wrist and the hold point), as the Genesis stop does:
            # never the touched box itself, which only the contact solver knows.
            points = np.array([self._wrist[arm].pos, self._hold_point(arm)])
            record_stop(self, guard, points, self._scene)
        guard.active = False
        self.moves.append(Move(arm, start, pos.copy(), tuple(stop_on), bump if stopped and touch is not None else None))
        if stopped:
            self.settle(0.2)
        elif dist > REACH_M and timeout_s > 0.5:
            self.settle(timeout_s - 0.5)
        return float(np.linalg.norm(pos - self._wrist[arm].pos))

    def set_hand(self, arm: Arm, closed: float) -> None:
        self.calls.append(f"set_hand:{arm}:{closed:.2f}")
        self._closure[arm] = float(np.clip(closed, 0.0, 1.0))
        self._update_hold(arm)

    def squeeze(self, arm: Arm, effort: float) -> None:
        self.calls.append(f"squeeze:{arm}:{effort:.2f}")
        self._closure[arm] = 1.0 if effort > 0 else 0.0
        self._update_hold(arm)

    def _update_hold(self, arm: Arm) -> None:
        assert self._scene is not None
        if self._closure[arm] >= 0.5:
            for name in self._scene.free_names:
                if name in self._held or name in self._welded:
                    continue
                if np.linalg.norm(self._objects[name] - self._hold_point(arm)) <= HAND_REACH_M:
                    self._held[name] = arm
        else:
            for name in [n for n, a in self._held.items() if a == arm]:
                del self._held[name]

    def hand_closure(self, arm: Arm) -> float:
        return self._closure[arm]

    def hand_contacts(self, arm: Arm, object_name: str) -> int:
        if self._held.get(object_name) == arm or self._welded.get(object_name) == arm:
            return 4
        if self._touch is not None and self._touch["arm"] == arm and self._touch["box"] == object_name:
            return 1
        near = np.linalg.norm(self._objects[object_name] - self._hold_point(arm)) <= 0.03
        return 1 if near else 0

    def attach(self, object_name: str, arm: Arm, link: str | None = None) -> None:  # the weld is to the wrist either way
        self.calls.append(f"attach:{object_name}")
        self._welded[object_name] = arm
        pose = self._wrist[arm]
        self._weld_offset[object_name] = _rotate(_conj(pose.quat), np.asarray(self._objects[object_name], dtype=float) - pose.pos)

    def detach(self, object_name: str) -> None:
        self.calls.append(f"detach:{object_name}")
        self._welded.pop(object_name, None)
        self._weld_offset.pop(object_name, None)

    # ----- joints, head -----
    def set_joints(self, targets: Mapping[str, float], *, immediate: bool = False) -> None:
        self._joints.update({k: float(v) for k, v in targets.items()})

    def joint_positions(self, names: tuple[str, ...] | None = None) -> dict[str, float]:
        if names is None:
            return dict(self._joints)
        return {n: self._joints.get(n, 0.0) for n in names}

    def set_head(self, yaw: float, pitch: float) -> None:
        self._head = (yaw, pitch)

    # ----- base -----
    def drive(self, linear_mps: float, angular_radps: float) -> None:
        self._drive = (float(linear_mps), float(angular_radps))

    def base_pose(self) -> Pose:
        x, y, yaw = self._base
        return Pose(np.array([x, y, 0.145]), quat_from_yaw(yaw))

    # ----- the robot's own sensors (scripted, deterministic) -----
    def inject_imu(self, placement: str, acc=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0), steps: int = 1) -> None:
        """Add ``acc`` (m/s^2) and ``gyro`` (rad/s) to that IMU's readings until ``steps`` physics steps have passed."""
        if placement not in dict(self.sensors.placements):
            raise KeyError(f"no IMU at {placement!r}; placements: {[p for p, _ in self.sensors.placements]}")
        self._spikes.append(ImuSpike(placement, np.asarray(acc, dtype=float), np.asarray(gyro, dtype=float), int(steps)))

    def read_imus(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        spec = self.sensors.imu
        out = {}
        for placement, _link in self.sensors.placements:
            # At rest an accelerometer reads the reaction to gravity, straight up; everything rides the base's turn.
            acc = np.array([0.0, 0.0, G0])
            gyro = np.array([0.0, 0.0, self._drive[1]])
            for spike in self._spikes:
                if spike.placement == placement:
                    acc = acc + spike.acc
                    gyro = gyro + spike.gyro
            out[placement] = (np.clip(acc, -spec.acc_range, spec.acc_range), np.clip(gyro, -spec.gyro_range, spec.gyro_range))
        return out

    def set_joint_torques(self, arm: Arm, torques) -> None:
        """Script what ``read_joint_torques(arm)`` returns (7 values, N*m) until the next reset."""
        values = np.asarray(torques, dtype=float)
        if values.shape != (7,):
            raise ValueError(f"an arm has 7 joints, got shape {values.shape}")
        self._torques[arm] = values

    def read_joint_torques(self, arm: Arm) -> np.ndarray:
        if arm not in self.sensors.arms("torque"):
            return np.zeros(0)
        torques = self._torques.get(arm, np.zeros(7)).copy()
        if self._touch is not None and self._touch["arm"] == arm:  # the motors hold against the box the hand presses on
            torques += min(self.TOUCH_TORQUE_RAMP_NM * self._touch["steps"], self.TOUCH_TORQUE_MAX_NM)
        return torques

    def set_tactile(self, arm: Arm, link: str, force) -> None:
        """Script one skin link's force xyz (N); the read clips its magnitude at the skin's range, as the part would."""
        if link not in skin_links(arm):
            raise KeyError(f"{link!r} is not a skin link of the {arm} hand: {skin_links(arm)}")
        self._tactile.setdefault(arm, {})[link] = np.asarray(force, dtype=float)

    def read_tactile(self, arm: Arm) -> dict[str, np.ndarray]:
        if arm not in self.sensors.arms("tactile"):
            return {}
        spec = self.sensors.skin
        scripted = dict(self._tactile.get(arm, {}))
        if self._touch is not None and self._touch["arm"] == arm:  # the touch presses on the palm
            palm = f"{arm}_gripper_base"
            scripted[palm] = scripted.get(palm, np.zeros(3)) + np.array([0.0, 0.0, self.TOUCH_PALM_N])
        return {link: spec.reading(scripted.get(link, np.zeros(3))) for link in skin_links(arm)}

    def set_fingertips(self, arm: Arm, forces: Mapping[str, float] | None) -> None:
        """Script the true force (N) on named fingertips, overriding the squeeze model for them; None clears it.
        The read still applies the FSR's actuation threshold and range."""
        if forces is None:
            self._fingertips.pop(arm, None)
            return
        unknown = set(forces) - set(fingertip_links(arm))
        if unknown:
            raise KeyError(f"not fingertips of the {arm} hand: {sorted(unknown)}")
        self._fingertips[arm] = {k: float(v) for k, v in forces.items()}

    def part_in_hand(self, arm: Arm) -> str | None:
        """The free part nearest the hand's hold point, if one is within hand reach (held or not)."""
        assert self._scene is not None
        hold = self._hold_point(arm)
        near = [(float(np.linalg.norm(self._objects[n] - hold)), n) for n in self._scene.free_names]
        near = [(d, n) for d, n in near if d <= HAND_REACH_M]
        return min(near)[1] if near else None

    def squeeze_depth(self, arm: Arm) -> float:
        """How far (m) the commanded fingertip gap sits inside the part in the hand's narrower side; 0 if no part or
        the fingers have not reached it."""
        assert self._scene is not None
        part = self.part_in_hand(arm)
        if part is None:
            return 0.0
        width = float(min(self._scene.box(part).size[:2]))
        return max(0.0, width - OPEN_APERTURE_M * (1.0 - self._closure[arm]))

    def read_fingertips(self, arm: Arm) -> dict[str, float]:
        if arm not in self.sensors.arms("fingertip"):
            return {}
        spec = self.sensors.fingertip
        squeeze = TIP_STIFFNESS_N_PER_M * self.squeeze_depth(arm)
        scripted = self._fingertips.get(arm, {})
        return {tip: spec.reading(scripted.get(tip, squeeze)) for tip in fingertip_links(arm)}

    def truth_hand_contacts(self, arm: Arm, boxes: tuple[str, ...]) -> dict[str, int]:
        return {name: self.hand_contacts(arm, name) for name in boxes}

    def set_finger_penetration(self, arm: Arm, part: str, depths: Mapping[str, float]) -> None:
        """Script ``truth_finger_penetration(arm, part)``: link name -> depth (m)."""
        self._penetration[(arm, part)] = {k: float(v) for k, v in depths.items()}

    def truth_finger_penetration(self, arm: Arm, part: str) -> dict[str, float]:
        return dict(self._penetration.get((arm, part), {}))

    # ----- sensing -----
    def capture(self, camera: str = "head") -> Frame:
        return Frame(rgb=np.zeros((30, 48, 3), dtype=np.uint8), depth=None)

    def capture_cloud(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """A stand-in cloud: the rim of every tray, in the tray's colour, where the tray is now."""
        assert self._scene is not None
        colors, points = [], []
        for box in self._scene.boxes:
            if not box.name.endswith("_floor"):
                continue
            wall = next(b for b in self._scene.boxes if b.name == box.name.replace("_floor", "_wall_xm"))
            cx, cy, _ = self._objects[box.name]
            half, top = box.size[0] / 2.0, float(self._objects[wall.name][2]) + wall.size[2] / 2.0
            for t in np.linspace(-half, half, 40):
                for x, y in ((cx + t, cy - half), (cx + t, cy + half), (cx - half, cy + t), (cx + half, cy + t)):
                    points.append((x, y, top))
                    colors.append([int(255 * c) for c in box.color])
        return np.array(colors, dtype=np.uint8)[None], np.array(points, dtype=float)[None]


__all__ = ["BoxSpec", "FakeSimulator", "ImuSpike", "Move"]


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the unit (w, x, y, z) quaternion ``q``."""
    q = np.asarray(q, dtype=float) / float(np.linalg.norm(q))
    return quat_mul(quat_mul(q, np.array([0.0, *v])), _conj(q))[1:]
