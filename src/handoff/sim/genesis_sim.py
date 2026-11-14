"""The Genesis implementation of :class:`Simulator`, built on the kit's ``MH3Simulation``.

This is the only module in ``handoff`` that imports Genesis or the kit. It adds the scene's
boxes and the robot's sensors (``config/sensors.yaml``: Genesis IMU, JointTorque and ContactForce
sensors with the part specs of ``base.py``) through the kit's scene hook, reads object state,
contacts and sensors from the native entities,
and leaves the kit's files untouched: the one thing the kit gives no hook for (rigid solver
options, needed for stable box-on-box contact) is set by wrapping ``gs.Scene`` for the
duration of the kit's own ``build``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mh3_genesis import MH3Simulation, SimulatorConfig
from mh3_genesis.simulator import _public_joint_name, _public_link_name, _quat_rotate  # the kit's own naming and math

from ..bump import BumpConfig, BumpGuard, record_stop, start_guard
from ..finish import CUBE_OBJ, capsule_obj, write_cube_obj
from .base import (
    DEFAULT_SENSORS,
    Arm,
    Frame,
    ObjectState,
    Pose,
    SceneSpec,
    SensorConfig,
    Simulator,
    fingertip_links,
    quat_from_yaw,
    skin_links,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "handoff.yaml"

WRIST_LINK = {"left": "left_gripper_base", "right": "right_gripper_base"}
ARM_JOINTS = {arm: tuple(f"{arm}_{i}" for i in range(1, 8)) for arm in ("left", "right")}
FINGER_JOINTS = {arm: tuple(f"{arm}_{f}" for f in ("index", "middle", "ring", "little")) for arm in ("left", "right")}
THUMB_JOINTS = {arm: (f"{arm}_thumb_1", f"{arm}_thumb_2") for arm in ("left", "right")}
# Fully closed targets. Fingers travel 0..1.57 rad; the thumb joints 0..1.6 and 0..0.45.
FINGER_CLOSED = 1.45
THUMB_CLOSED = (1.2, 0.4)
CAMERAS = {"head": "head_left", "base": "base_left"}
# Finger PD gains. The kit's body default (350 / 35) with no effort limit makes a closed hand press
# with about 200 N and jitter the part out; measured Oct 11: kp 5 held 9/10, kp 350 7/10 (4 cm part).
HAND_KP, HAND_KV = 5.0, 0.5
# A wrist move follows the straight line to its target: IK every LINE_STEP_M along it, each seeded with the last, and
# the joints interpolated between those points. Interpolating the joints straight from start to goal (as before Oct 21)
# swings the hand on an arc: a 45 cm crossing at 1.03 m sagged to 0.945 m and dragged the hand over both trays' wall
# tops (scripts/probe_crossing_trace.py, seed 7, Genesis CPU), so the bump guard stopped every carry.
LINE_STEP_M = 0.01


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation of unit quaternions (w, x, y, z), the short way round."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


# The IK rot_mask that frees yaw. Genesis 1.4.3 masks the rotation error rotvec(target * inv(hand)), which is in WORLD
# axes, and inverts a mask with one True axis (rigid_entity.inverse_kinematics: "align 1 axis"): [False, False, True]
# holds the world-x and world-y error and frees the rotation about world z, so the hand stays level and turns about
# vertical. The first version (1d0a0e8) passed the hand's own up axis, [False, True, False] for the grasp orientation,
# which froze the yaw and freed the pitch about world y instead (found by the Oct 21 review).
YAW_FREE_MASK = [False, False, True]
# A step in the line's joint path larger than this (rad, any joint, between points 1 cm apart) is an IK branch flip,
# not motion: the move falls back to the joint path straight to the goal's IK rather than whip through it.
LINE_JUMP_RAD = 0.25


def along(path: np.ndarray, s: float) -> np.ndarray:
    """The point a fraction ``s`` (0..1) along a sequence of equally spaced joint vectors, linear between them."""
    m = len(path) - 1
    x = min(max(s, 0.0), 1.0) * m
    i = min(int(x), m - 1) if m > 0 else 0
    return path[i] + (x - i) * (path[min(i + 1, m)] - path[i])


class _TrialSimulation(MH3Simulation):
    """The kit's simulation with solver options the kit does not expose: box-box detection, and ``substeps`` physics
    substeps per step (each control step of dt stays the same; the solver integrates ``substeps`` times inside it).

    Substeps are what make contact stiffer: Genesis floors a contact's time constant at twice the integration interval
    (RigidOptions.constraint_timeconst), so at dt 0.01 and one substep every contact responds over 0.02 s and a
    squeezed part is visibly sunk into the fingers; 2 substeps halve the interval, 4 quarter it."""

    substeps: int = 1

    def build(self) -> MH3Simulation:
        import genesis as gs

        original_scene = gs.Scene
        substeps = self.substeps

        def scene_with_rigid_options(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("rigid_options", gs.options.RigidOptions(box_box_detection=True))
            options = kwargs.get("sim_options")
            if substeps > 1 and options is not None:
                kwargs["sim_options"] = gs.options.SimOptions(dt=options.dt, gravity=options.gravity, substeps=substeps)
            return original_scene(*args, **kwargs)

        gs.Scene = scene_with_rigid_options
        try:
            return super().build()
        finally:
            gs.Scene = original_scene


class GenesisSimulator(Simulator):
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        hand_gains: tuple[float, float] = (HAND_KP, HAND_KV),
        sensors: SensorConfig | str | Path | None = DEFAULT_SENSORS,
        substeps: int = 1,
        help_text: bool = True,
        line_step_m: float | None = None,
    ) -> None:
        """``sensors``: a SensorConfig, a sensors YAML (``config/sensors.yaml`` by default, on), or None for none.
        ``substeps``: physics substeps per step, for stiffer contact (see _TrialSimulation).
        ``help_text``: Genesis's key list on the viewer (78.8 s of a 465 s capsule shift, profile_shift.py, Nov 5).
        ``line_step_m``: how far apart move_wrist solves IK along a straight line (default LINE_STEP_M); each solve is a
        GPU round trip of 90 to 210 ms (same profile)."""
        self._help_text = bool(help_text)
        self.line_step_m = float(line_step_m) if line_step_m is not None else LINE_STEP_M
        self._config = SimulatorConfig.from_yaml(config_path)
        self._substeps = int(substeps)
        if self._substeps < 1:
            raise ValueError(f"substeps must be at least 1, not {substeps}")
        if sensors is None:
            self.sensors = SensorConfig.off()
        elif isinstance(sensors, SensorConfig):
            self.sensors = sensors
        else:
            self.sensors = SensorConfig.from_yaml(sensors)
        self._imus: dict[str, Any] = {}  # placement -> Genesis IMU sensor
        self._torque: dict[str, Any] = {}  # arm -> Genesis JointTorque sensor over its 7 joints
        self._skin: dict[str, dict[str, Any]] = {}  # arm -> skin link -> Genesis ContactForce sensor
        self._tips: dict[str, dict[str, tuple[Any, Any]]] = {}  # arm -> tip -> (tip link, pad link) ContactForce
        self._tip_gain: dict[str, float] = {}  # tip -> the FSR's part-to-part gain, drawn once per build
        self._pad_in_tip: dict[str, np.ndarray] = {}  # tip -> pad link orientation in the tip link frame (w, x, y, z)
        self._noise_rng = np.random.default_rng([self.sensors.seed, 31])
        self._tip_cache: dict[str, tuple[int, dict[str, float]]] = {}  # arm -> (step it was read at, readings)
        self._skin_cache: dict[str, tuple[int, dict[str, np.ndarray]]] = {}  # the same for the skin
        self._steps = 0  # physics steps since build, never reset: keys the skin and fingertip noise to one draw per step
        self._link_names: dict[int, str] = {}  # global link index -> the kit's public name
        self._hand_gains = hand_gains
        self._sim: MH3Simulation | None = None
        self._scene: SceneSpec | None = None
        self._entities: dict[str, Any] = {}
        self._hand_links: dict[str, set[int]] = {}
        self._welds: dict[str, tuple[int, int]] = {}
        self._brake: tuple[int, int] | None = None  # (base link, anchor link) while the base is held
        self._scene_cams: dict[str, Any] = {}
        self._t = 0.0

    @property
    def kit(self) -> MH3Simulation:
        """The kit's own simulation object, for the control panel and anything Genesis-native."""
        assert self._sim is not None, "build() first"
        return self._sim

    # ----- world -----
    @property
    def dt(self) -> float:
        return self._config.dt

    @property
    def time(self) -> float:
        return self._t

    def build(self, scene: SceneSpec) -> None:
        self._scene = scene
        sim = _TrialSimulation(self._config)
        sim.substeps = self._substeps
        if not self._help_text:
            sim.extra_viewer = {"enable_help_text": False}
        if not scene.kit_floor:
            sim.floor_visible = False
        if scene.lamps or scene.ambient is not None:
            import genesis as gs

            sun = [gs.options.vis.DirectionalLight(dir=(-1, -1, -1), color=(1.0, 1.0, 1.0), intensity=scene.sun)] if scene.sun > 0 else []  # the kit's default 5.0
            lamps = [gs.options.vis.PointLight(pos=(x, y, z), color=(1.0, 1.0, 1.0), intensity=i) for x, y, z, i in scene.lamps]
            sim.extra_vis = {"lights": [*sun, *lamps], **({"ambient_light": (scene.ambient,) * 3} if scene.ambient is not None else {})}

        def add_scene(simulation: MH3Simulation) -> None:
            gs = simulation.gs
            if not CUBE_OBJ.exists():
                write_cube_obj()
            looks = {"plastic": gs.surfaces.Plastic, "matte": gs.surfaces.Rough, "glossy": gs.surfaces.Smooth}
            for box in scene.boxes:
                material = gs.materials.Rigid(rho=box.rho, friction=box.friction)
                euler = (0.0, 0.0, box.yaw_deg)
                if box.texture is not None:
                    # Genesis boxes carry no texture coordinates; a textured part is a UV cube mesh.
                    morph = gs.morphs.Mesh(file=str(CUBE_OBJ), scale=box.size, pos=box.pos, euler=euler, fixed=box.fixed, collision=box.collision)
                    surface = gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_array=box.texture))
                else:
                    if box.shape == "capsule":
                        morph = gs.morphs.Mesh(file=str(capsule_obj(box.size[0], box.size[1], box.size[2])), pos=box.pos, euler=euler, fixed=box.fixed, collision=box.collision)
                    elif box.shape == "disc":
                        morph = gs.morphs.Cylinder(pos=box.pos, radius=box.size[0] / 2.0, height=box.size[2], fixed=box.fixed, collision=box.collision)
                    else:
                        morph = gs.morphs.Box(pos=box.pos, size=box.size, euler=euler, fixed=box.fixed, collision=box.collision)
                    surface = looks[box.look](color=box.color)
                self._entities[box.name] = simulation.scene.add_entity(morph, material=material, surface=surface)
            for cam in scene.cameras:
                self._scene_cams[cam.name] = simulation.scene.add_camera(res=cam.res, pos=cam.pos, lookat=cam.lookat, fov=cam.fov, GUI=False)
            self._add_sensors(simulation)

        sim.add_scene_hook(add_scene)
        sim.build()
        self._sim = sim
        for arm in ("left", "right"):
            names = [n for n in sim.link_names if n.startswith(f"{arm}_") and not n.startswith(f"{arm}_arm_")]
            self._hand_links[arm] = {int(sim.get_link(n).idx) for n in names}
            self.set_hand_gains(arm, *self._hand_gains)
        self._link_names = {int(sim.get_link(n).idx): n for n in sim.link_names}
        for arm, tips in self._tips.items():
            for tip, (tip_link, pad_link) in fingertip_links(arm).items():  # type: ignore[arg-type]
                if tip in tips:  # the pad is welded to the tip: its orientation in the tip frame never changes
                    self._pad_in_tip[tip] = np.asarray(sim.get_link_pose(pad_link, frame=tip_link)[1], dtype=float)
        if not self._help_text:
            # Genesis sets the key-press message's fade only with the help text on; a key pressed in the viewer then
            # raised AttributeError on every press (dup trying the keys, Nov 6). The message still shows and fades.
            viewer = getattr(getattr(sim.scene, "viewer", None), "_pyrender_viewer", None)
            if viewer is not None and not hasattr(viewer, "_ticks_till_fade"):
                viewer._message_text = None
                viewer._ticks_till_fade = 2.0 / 3.0 * viewer.viewer_flags["refresh_rate"]
                viewer._message_opac = 1.0 + viewer._ticks_till_fade
        self._after_reset()

    def _add_sensors(self, simulation: MH3Simulation) -> None:
        """Put the configured sensors on the robot. Runs inside the kit's scene hook: Genesis takes sensors only
        before ``scene.build()``, and the robot entity exists by then. The noise is the part specs' (``base.py``)."""
        cfg = self.sensors
        if not cfg.enabled:
            return
        gs, robot, add = simulation.gs, simulation.robot, simulation.scene.add_sensor
        # The kit's public names, built the way the kit builds them after the scene (its lookups need a built scene).
        links = {_public_link_name(link.name.rstrip("/").rsplit("/", 1)[-1]): link for link in robot.links}

        def local(name: str) -> int:
            if name not in links:
                raise KeyError(f"no robot link {name!r} to put a sensor on; links: {sorted(links)}")
            return int(links[name].idx_local)

        acc_sigma, gyro_sigma = cfg.imu.sigma(self.dt)
        for placement, link in cfg.placements:
            self._imus[placement] = add(gs.sensors.IMU(
                entity_idx=robot.idx, link_idx_local=local(link),
                acc_noise=acc_sigma, gyro_noise=gyro_sigma,
                acc_resolution=cfg.imu.acc_resolution, gyro_resolution=cfg.imu.gyro_resolution,
            ))
        dof_of = {
            _public_joint_name(joint.name.strip().rstrip("/").rsplit("/", 1)[-1]): int(joint.dofs_idx_local[0])
            for joint in robot.joints
            if joint.n_dofs == 1
        }
        for arm in cfg.arms("torque"):
            self._torque[arm] = add(gs.sensors.JointTorque(
                entity_idx=robot.idx, dofs_idx_local=[dof_of[name] for name in ARM_JOINTS[arm]],
                noise=cfg.torque.noise, resolution=cfg.torque.resolution,
            ))
        for arm in cfg.arms("tactile"):
            # Noise-free contact force; the skin's proportional noise and range (SkinSpec.reading) are applied per read.
            self._skin[arm] = {
                link: add(gs.sensors.ContactForce(entity_idx=robot.idx, link_idx_local=local(link)))
                for link in skin_links(arm)  # type: ignore[arg-type]
            }
        gain_rng = np.random.default_rng([cfg.seed, 4242])
        for arm in cfg.arms("fingertip"):
            # Noise-free contact force on the tip and its pad; the FSR model (FingertipSpec.reading) is applied per read.
            self._tips[arm] = {
                tip: tuple(add(gs.sensors.ContactForce(entity_idx=robot.idx, link_idx_local=local(link))) for link in pair)
                for tip, pair in fingertip_links(arm).items()  # type: ignore[arg-type]
            }
            for tip in self._tips[arm]:
                self._tip_gain[tip] = float(1.0 + cfg.fingertip.part_to_part * gain_rng.standard_normal())

    def _after_reset(self) -> None:
        sim = self.kit
        # The kit applies its home pose after build, so a scene reset lands on the USD pose; re-apply.
        sim.set_joint_positions(sim.home_positions, immediate=True)
        sim.set_joint_positions(sim.home_positions)
        # Brake the base: wheels are free-rolling until they get a velocity command.
        sim.stop_base()
        self._t = 0.0
        self.step(5)

    def reset(self) -> None:
        sim = self.kit
        self.clear_controllers()
        for name in list(self._welds):
            self.detach(name)
        self.brake(False)
        sim.scene.reset()
        self._after_reset()

    def step(self, n: int = 1) -> None:
        sim = self.kit
        for _ in range(n):
            # Counted before the kit steps, so a kit step hook (the demo's panel windows) that reads a sensor inside a
            # multi-step call already sees this step's count and gets this step's reading, not the cached one before.
            self._steps += 1
            sim.step()
            self._run_controllers()
        self._t += n * self.dt

    # ----- objects -----
    @staticmethod
    def _np(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=float)

    def object_state(self, name: str) -> ObjectState:
        ent = self._entities[name]
        return ObjectState(self._np(ent.get_pos()), self._np(ent.get_quat()), self._np(ent.get_vel()))

    def set_object_pose(self, name: str, pos: tuple[float, float, float], yaw_deg: float = 0.0) -> None:
        ent = self._entities[name]
        ent.set_pos(np.asarray(pos, dtype=float), zero_velocity=True)
        ent.set_quat(quat_from_yaw(np.deg2rad(yaw_deg)), zero_velocity=True)

    # ----- arms and hands -----
    def wrist_pose(self, arm: Arm) -> Pose:
        pos, quat = self.kit.get_link_pose(WRIST_LINK[arm])
        return Pose(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))

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
        sim = self.kit
        pos = np.asarray(pos, dtype=float)
        joints = ARM_JOINTS[arm]
        self.last_bump = None
        guard = self._start_guard(arm, stop_on)  # the bump detectors of this move (config/bump.yaml), None when unguarded
        # The kit maps public joint names to qpos indices privately; read-only use of that map.
        q_idx = sim._public_q_indices  # noqa: SLF001
        here = self.wrist_pose(arm)
        start = np.array([self.joint_positions(joints)[name] for name in joints])
        dist = float(np.linalg.norm(here.pos - pos))
        m = max(1, int(np.ceil(dist / self.line_step_m)))
        path, seed = [start], {}
        mask = {"rot_mask": YAW_FREE_MASK} if free_yaw else {}
        for k in range(1, m + 1):  # IK along the straight line, each point seeded with the one before
            solution = sim.solve_ik(WRIST_LINK[arm], position=here.pos + (k / m) * (pos - here.pos), orientation=slerp(here.quat, quat, k / m),
                                    joints=joints, apply=False, **seed, **mask)
            seed = {"init_qpos": solution}
            path.append(np.array([float(solution[q_idx[name]]) for name in joints]))
        path = np.array(path)
        jump = float(np.abs(np.diff(path, axis=0)).max()) if m > 1 else 0.0
        if jump > LINE_JUMP_RAD:  # a branch flip mid-line: no whip
            # Every point after the flip was seeded from the solver's random restart, the goal's included; the goal
            # solved fresh from the start joints (Genesis seeds from the current pose) is on the branch the arm is on.
            # Whichever of the two is nearer the start joints is where the joint path goes (Oct 22 nightly review).
            fresh = sim.solve_ik(WRIST_LINK[arm], position=pos, orientation=quat, joints=joints, apply=False, **mask)
            fresh = np.array([float(fresh[q_idx[name]]) for name in joints])
            chained = path[-1]
            use_fresh = float(np.abs(fresh - start).max()) <= float(np.abs(chained - start).max())
            goal = fresh if use_fresh else chained
            self.line_fallbacks = getattr(self, "line_fallbacks", 0) + 1
            if self.line_fallback_log is None:
                self.line_fallback_log = []
            self.line_fallback_log.append({"t": round(self._t, 3), "arm": arm, "target": [round(float(v), 3) for v in pos], "jump_rad": round(jump, 3),
                                           "goal": "fresh" if use_fresh else "chained", "swing_rad": round(float(np.abs(goal - start).max()), 3)})
            path = np.array([start, goal])
        if duration_s is None:
            duration_s = float(np.clip(dist / 0.25, 0.3, 2.0))  # about 0.25 m/s, eased
        n = max(1, int(round(duration_s / self.dt)))
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * k / n)  # ease in and out: no acceleration spikes
            sim.set_joint_positions(dict(zip(joints, along(path, s), strict=True)))
            self.step(1)
            if guard is not None and self._stopped_on(arm, guard):
                return float(np.linalg.norm(self.wrist_pose(arm).pos - pos))
        err = float("inf")
        settled_checks = 0
        check_every = 5
        for _ in range(int(timeout_s / self.dt / check_every)):
            for _ in range(check_every if guard is not None else 1):  # a guarded move feeds its detectors every step
                self.step(1 if guard is not None else check_every)
                if guard is not None and self._stopped_on(arm, guard):
                    return float(np.linalg.norm(self.wrist_pose(arm).pos - pos))
            err = float(np.linalg.norm(self.wrist_pose(arm).pos - pos))
            settled_checks = settled_checks + 1 if err <= tol_m else 0
            if settled_checks >= 3:
                break
        if guard is not None:
            guard.active = False
        return err

    # ----- bump detection: the robot's own sensors (handoff.bump), never the contact solver unless asked for as truth -----
    bump_config: BumpConfig | None = None  # None: config/bump.yaml, read on the first guarded move
    bump_guard: BumpGuard | None = None  # the last guarded move's detectors, for the panel's windows
    bump_log: list[dict[str, Any]] | None = None  # every bump detected since build, as BumpGuard.report dicts
    line_fallback_log: list[dict[str, Any]] | None = None  # every straight line move_wrist gave up for the joint path
    bins_seen: Mapping[str, tuple[float, float]] | None = None  # the task's bins_seen: where the robot saw each tray
    _vel_seen: dict[int, np.ndarray] | None = None  # physics step -> robot dof velocity, the last two steps read
    _model_cache: dict[str, tuple[int, Any]] | None = None  # arm -> (step, torque_model result)
    _arm_dof_idx: dict[str, list[int]] | None = None  # arm -> robot-local dof of each arm joint
    last_bump_info: dict[str, Any] | None = None  # the last detected bump's report (BumpGuard.report)

    def _start_guard(self, arm: Arm, stop_on: tuple[str, ...]) -> BumpGuard | None:
        return start_guard(self, arm, stop_on)

    def _stopped_on(self, arm: Arm, guard: BumpGuard) -> bool:
        """Feed the move's bump detectors one step; if the configured one fires, hold the arm where it is, name the box
        (an estimate: the detectors cannot tell which box) in ``last_bump``, log it in ``bump_log``, and return True."""
        if not guard.step():
            return False
        self.kit.set_joint_positions(self.joint_positions(ARM_JOINTS[arm]))  # stop pushing into it
        points = np.array([self.kit.get_link_pose(name)[0] for name in self._hand_link_names(arm)], dtype=float)
        record_stop(self, guard, points, self._scene)
        return True

    def _hand_link_names(self, arm: Arm) -> list[str]:
        """The wrist and every link of that hand, by the kit's public names: where the arm's kinematics put the hand."""
        return [WRIST_LINK[arm], *(n for n in self.kit.link_names if n.startswith(f"{arm}_") and not n.startswith(f"{arm}_arm_") and n != WRIST_LINK[arm])]

    def torque_model(self, arm: Arm) -> tuple[np.ndarray, np.ndarray] | None:
        """The robot's dynamics model over the last physics step, the arm's 7 joints, for the momentum observer:
        ``(dp, bias)``. ``dp = M (qdot_after - qdot_before)`` with the mass matrix the step integrated with (Genesis
        folds the motor armature into it; the torque reading already takes the armature's share out, so it is taken
        out of M too) and ``bias`` the gravity and Coriolis torque of that step. None until two consecutive steps
        have been read. This is the robot's MODEL (in a real cell, from its URDF), not the contact solver: it says
        nothing about what touches what."""
        step = self._steps
        if self._model_cache is None:
            self._model_cache = {}
        cached = self._model_cache.get(arm)
        if cached is not None and cached[0] == step:
            return cached[1]
        robot = self.kit.robot
        if self._vel_seen is None or step not in self._vel_seen:
            seen = {} if self._vel_seen is None else {k: v for k, v in self._vel_seen.items() if k == step - 1}
            seen[step] = self._np(robot.get_dofs_velocity()).reshape(-1)
            self._vel_seen = seen
        before = self._vel_seen.get(step - 1)
        result = None
        if before is not None:
            from genesis.utils.misc import qd_to_torch

            dofs = self._arm_dofs(arm)
            mass = self._np(robot.get_mass_mat()) - np.diag(self._np(robot.get_dofs_armature()).reshape(-1))
            solver = self.kit.scene.sim.rigid_solver
            bias = self._np(qd_to_torch(solver.dyn_state.dofs.qf_bias, transpose=True)).reshape(-1)
            bias = bias[robot.dof_start : robot.dof_start + robot.n_dofs]
            result = (mass[dofs] @ (self._vel_seen[step] - before), bias[dofs])
        self._model_cache[arm] = (step, result)
        return result

    def _arm_dofs(self, arm: Arm) -> list[int]:
        """The robot-local dof index of each of that arm's 7 joints, ``<arm>_1`` .. ``<arm>_7``."""
        if self._arm_dof_idx is None:
            self._arm_dof_idx = {}
        if arm not in self._arm_dof_idx:
            dof_of = {
                _public_joint_name(joint.name.strip().rstrip("/").rsplit("/", 1)[-1]): int(joint.dofs_idx_local[0])
                for joint in self.kit.robot.joints
                if joint.n_dofs == 1
            }
            self._arm_dof_idx[arm] = [dof_of[name] for name in ARM_JOINTS[arm]]
        return self._arm_dof_idx[arm]

    def set_hand(self, arm: Arm, closed: float) -> None:
        closed = float(np.clip(closed, 0.0, 1.0))
        targets = {j: closed * FINGER_CLOSED for j in FINGER_JOINTS[arm]}
        targets[THUMB_JOINTS[arm][0]] = closed * THUMB_CLOSED[0]
        targets[THUMB_JOINTS[arm][1]] = closed * THUMB_CLOSED[1]
        self.kit.set_joint_positions(targets)

    def set_thumb(self, arm: Arm, closed: float) -> bool:
        closed = float(np.clip(closed, 0.0, 1.0))
        self.kit.set_joint_positions({THUMB_JOINTS[arm][0]: closed * THUMB_CLOSED[0], THUMB_JOINTS[arm][1]: closed * THUMB_CLOSED[1]})
        return True

    def thumb_closure(self, arm: Arm) -> float:
        state = self.kit.get_joint_state([THUMB_JOINTS[arm][0]])
        return float(np.clip(state.positions[0] / THUMB_CLOSED[0], 0.0, 1.0))

    def squeeze(self, arm: Arm, effort: float) -> None:
        self.kit.set_joint_efforts({j: float(effort) for j in FINGER_JOINTS[arm]})

    def set_hand_gains(self, arm: Arm, kp: float, kv: float) -> None:
        """Set PD gains on every finger and thumb joint of that hand, followers included.

        The kit gives the fingers the body default (kp 350) and no effort limit, so a closed hand
        presses with hundreds of newtons; a softer hand holds without crushing or jittering.
        """
        kit = self.kit
        usd_prefix = "L_" if arm == "left" else "R_"
        # Root finger joints are public; follower (mimic) joints are only reachable by USD name.
        indices = [kit._public_joint_indices[j] for j in FINGER_JOINTS[arm] + THUMB_JOINTS[arm]]  # noqa: SLF001
        indices += [kit._joint_indices[n] for n in kit._finger_mimics if n.startswith(usd_prefix)]  # noqa: SLF001
        kit.robot.set_dofs_kp(np.full(len(indices), float(kp)), indices)
        kit.robot.set_dofs_kv(np.full(len(indices), float(kv)), indices)

    def hand_closure(self, arm: Arm) -> float:
        state = self.kit.get_joint_state(FINGER_JOINTS[arm])
        return float(np.clip(np.mean(state.positions) / FINGER_CLOSED, 0.0, 1.0))

    def hand_contacts(self, arm: Arm, object_name: str) -> int:
        contacts = self.kit.robot.get_contacts(with_entity=self._entities[object_name])
        link_a = contacts.get("link_a")
        link_b = contacts.get("link_b")
        if link_a is None or link_b is None:
            return 0
        a = self._np(link_a).astype(int).ravel()
        b = self._np(link_b).astype(int).ravel()
        hand = self._hand_links[arm]
        return int(sum(1 for la, lb in zip(a, b, strict=True) if la in hand or lb in hand))

    def attach(self, object_name: str, arm: Arm, link: str | None = None) -> None:
        if object_name in self._welds:
            return
        ent = self._entities[object_name]
        part_link = int(ent.links[0].idx)
        wrist_link = int(self.kit.get_link(link or WRIST_LINK[arm]).idx)
        self.kit.scene.sim.rigid_solver.add_weld_constraint(part_link, wrist_link)
        self._welds[object_name] = (part_link, wrist_link)

    def brake(self, on: bool, anchor: str | None = None) -> None:
        """A weld from the mobile base to a fixed box stands in for locked wheels: the arms then cannot shove the base
        (Nov 5: the wrist pressing on a tray wall pushed it back 4 mm a reach)."""
        solver = self.kit.scene.sim.rigid_solver
        if self._brake is not None:
            solver.delete_weld_constraint(*self._brake)
            self._brake = None
        if on and anchor is not None:
            pair = (int(self.kit.get_link("mobile_base").idx), int(self._entities[anchor].links[0].idx))
            solver.add_weld_constraint(*pair)
            self._brake = pair

    def detach(self, object_name: str) -> None:
        pair = self._welds.pop(object_name, None)
        if pair is not None:
            self.kit.scene.sim.rigid_solver.delete_weld_constraint(*pair)

    # ----- joints, head -----
    def set_joints(self, targets: Mapping[str, float], *, immediate: bool = False) -> None:
        self.kit.set_joint_positions(dict(targets), immediate=immediate)

    def joint_positions(self, names: tuple[str, ...] | None = None) -> dict[str, float]:
        state = self.kit.get_joint_state(names)
        return {n: float(p) for n, p in zip(state.names, state.positions, strict=True)}

    def set_head(self, yaw: float, pitch: float) -> None:
        self.kit.set_joint_positions({"head_yaw": yaw, "head_pitch": pitch})

    # ----- base -----
    def drive(self, linear_mps: float, angular_radps: float) -> None:
        self.kit.drive_base(linear_mps, angular_radps)

    def base_pose(self) -> Pose:
        pos, quat = self.kit.get_link_pose("mobile_base")
        return Pose(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))

    # ----- sensing -----
    def capture(self, camera: str = "head") -> Frame:
        if camera in self._scene_cams:
            rgb, _, _, _ = self._scene_cams[camera].render(rgb=True, depth=False)
            return Frame(rgb=np.asarray(self._np(rgb), dtype=np.uint8), depth=None)
        frame = self.kit.capture_camera(CAMERAS.get(camera, camera), rgb=True, depth=False)
        rgb = None if frame.rgb is None else np.asarray(frame.rgb)
        return Frame(rgb=rgb, depth=None)

    def capture_cloud(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        cam = self._scene_cams[camera]
        rgb, _, _, _ = cam.render(rgb=True, depth=False)
        points, _ = cam.render_pointcloud(world_frame=True)
        return np.asarray(self._np(rgb), dtype=np.uint8), np.asarray(self._np(points), dtype=float)

    def tray_load(self, boxes: tuple[str, ...]) -> float:
        # The solver's net contact force on each box; a fixed box resting on the fixed table has no contact with it,
        # so what is left is what the parts (and a hand, if one touches) press on it. Down is -z.
        return float(sum(-self._np(self._entities[name].get_links_net_contact_force())[..., 2].sum() for name in boxes))

    # ----- the robot's own sensors: Genesis computes each one every step; a read is a view of the last one -----
    def read_imus(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        spec = self.sensors.imu
        out = {}
        for placement, sensor in self._imus.items():
            data = sensor.read()
            acc = np.clip(self._np(data.lin_acc).reshape(3), -spec.acc_range, spec.acc_range)
            gyro = np.clip(self._np(data.ang_vel).reshape(3), -spec.gyro_range, spec.gyro_range)
            out[placement] = (acc, gyro)
        return out

    def read_joint_torques(self, arm: Arm) -> np.ndarray:
        sensor = self._torque.get(arm)
        return np.zeros(0) if sensor is None else self._np(sensor.read()).reshape(-1)

    def read_tactile(self, arm: Arm) -> dict[str, np.ndarray]:
        skin = self._skin.get(arm)
        if not skin:
            return {}
        cached = self._skin_cache.get(arm)
        if cached is not None and cached[0] == self._steps:
            return {link: f.copy() for link, f in cached[1].items()}  # one noise draw per step, however often it is read
        spec = self.sensors.skin
        out = {link: spec.reading(self._np(s.read()).reshape(3), self._noise_rng.standard_normal(3)) for link, s in skin.items()}
        self._skin_cache[arm] = (self._steps, out)
        return {link: f.copy() for link, f in out.items()}

    def read_fingertips(self, arm: Arm) -> dict[str, float]:
        tips = self._tips.get(arm)
        if not tips:
            return {}
        cached = self._tip_cache.get(arm)
        if cached is not None and cached[0] == self._steps:
            return dict(cached[1])  # one noise draw per step, however often it is read
        spec = self.sensors.fingertip
        out = {}
        for tip, (tip_sensor, pad_sensor) in tips.items():
            # Each force is in its own link's frame; the pad's turns into the tip's by the fixed joint between them. The
            # magnitude of the sum stands in for the pad-normal force an FSR reads (FingertipSpec says why).
            pad = _quat_rotate(self._pad_in_tip[tip], self._np(pad_sensor.read()).reshape(3))
            force = self._np(tip_sensor.read()).reshape(3) + pad
            out[tip] = spec.reading(float(np.linalg.norm(force)), self._tip_gain[tip], float(self._noise_rng.standard_normal()))
        self._tip_cache[arm] = (self._steps, out)
        return dict(out)

    # ----- truth: the contact solver's answers, for checks and scoring only -----
    def truth_hand_contacts(self, arm: Arm, boxes: tuple[str, ...]) -> dict[str, int]:
        counts = dict.fromkeys(boxes, 0)
        owner = {int(link.idx): name for name in boxes for link in self._entities[name].links}
        contacts = self.kit.robot.get_contacts()
        link_a, link_b = contacts.get("link_a"), contacts.get("link_b")
        if link_a is None or link_b is None:
            return counts
        hand = self._hand_links[arm]
        for la, lb in zip(self._np(link_a).astype(int).ravel(), self._np(link_b).astype(int).ravel(), strict=True):
            if la in hand and lb in owner:
                counts[owner[lb]] += 1
            elif lb in hand and la in owner:
                counts[owner[la]] += 1
        return counts

    def truth_finger_penetration(self, arm: Arm, part: str) -> dict[str, float]:
        contacts = self.kit.robot.get_contacts(with_entity=self._entities[part])
        link_a, link_b, depth = contacts.get("link_a"), contacts.get("link_b"), contacts.get("penetration")
        if link_a is None or link_b is None or depth is None:
            return {}
        hand = self._hand_links[arm]
        out: dict[str, float] = {}
        a = self._np(link_a).astype(int).ravel()
        b = self._np(link_b).astype(int).ravel()
        for la, lb, d in zip(a, b, self._np(depth).ravel(), strict=True):
            link = la if la in hand else (lb if lb in hand else None)
            if link is not None:
                name = self._link_names[int(link)]
                out[name] = max(out.get(name, 0.0), float(d))
        return out
