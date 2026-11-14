"""The simulator interface.

Everything above this file (scene layout, skills, checks, trials, and later the behaviour
tree and the decision ladder) talks to the robot and the world only through ``Simulator``.
``genesis_sim.py`` is the Genesis implementation; an Unreal one would be a second file.

Robotics vocabulary used throughout:

- **pose**: a position (x, y, z in metres, world frame) plus an orientation. Orientations are
  unit quaternions in (w, x, y, z) order, the convention Genesis and the kit use.
- **wrist**: the root link of a hand (``left_gripper_base`` on the MH3). Arm targets are wrist
  poses; the fingers hang off the wrist.
- **IK, inverse kinematics**: given a wanted wrist pose, solve for the arm joint angles that put
  the wrist there. The joints are then driven to those angles by PD position controllers, so
  the arm takes some simulated time to arrive; ``move_wrist`` waits for that.
- **settle**: run physics for a while so controllers reach their targets and objects stop moving.
- **attach / detach**: a rigid "weld" between an object and the hand. In Genesis it is a runtime
  weld constraint, in Unreal it would be a socket attach. It is the comparison arm for the real
  finger grasp, never the default.
- **sensor vs truth**: a ``read_*`` method is what a real part on the robot would report, with that
  part's datasheet noise, resolution and range (the specs below cite each one). A ``truth_*``
  method (and ``hand_contacts``) asks the simulator's own contact solver: it exists for checks,
  scoring and tests, and decision logic must never read it, because no real robot can.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Arm = Literal["left", "right"]
G0 = 9.80665  # standard gravity, m/s^2: converts the datasheets' g and mg
# The base turns about its drive-wheel axle, which is this far ahead of the base origin (the wheels' collision boxes
# span x +0.10 to +0.30, probe_body.py, Nov 4): a turn in place swings the origin round that point.
AXLE_AHEAD = 0.20

# ----- the robot's sensors: which part, where, and that part's published numbers -----
#
# The specs hold only numbers a vendor (or, for joint torque, a measured study) published; what a page does not give
# (IMU bias stability and random walk, FSR hysteresis as a model) is left out and says so, never filled in.

# The kit's public link names of one hand: the palm and the phalanges carry the tactile skin, each fingertip a force
# sensor. The MH3 USD models that fingertip sensor as its own pad link fixed to the tip (``<arm>_<finger>_force_sensor``),
# so a fingertip reading is the contact force on the tip link plus its pad.
FINGERS = ("index", "middle", "ring", "little", "thumb")


def skin_links(arm: Arm) -> tuple[str, ...]:
    """The links one hand's tactile skin covers: the palm (``<arm>_gripper_base``) and every finger and thumb segment
    short of the tip."""
    return (f"{arm}_gripper_base", *(f"{arm}_{f}_{i}" for f in FINGERS for i in (1, 2)))


def fingertip_links(arm: Arm) -> dict[str, tuple[str, str]]:
    """Each fingertip sensor by its tip link name, with the two links whose contact force it reads (tip, pad)."""
    return {f"{arm}_{f}_tip": (f"{arm}_{f}_tip", f"{arm}_{f}_force_sensor") for f in FINGERS}


@dataclass(frozen=True)
class ImuSpec:
    """Stereolabs ZED X IMU, the same part at every placement (the MH3's head camera is a ZED X Mini and its base
    camera a ZED X; the page's "Motion sensor" table covers both). Source, read Oct 21:
    https://docs.stereolabs.com/docs/products/cameras/zedx/specifications.md

    The page gives noise density as 2.3 mg and 0.20 dps with no unit of time. Stereolabs' own SDK defines its
    ``noise_density`` as continuous white noise in ``sensor_unit / sqrt(Hz)``
    (https://stereolabs.com/docs/api/structsl_1_1SensorParameters.html), so it is read that way here, and one sample's
    standard deviation is density * sqrt(sample rate) (the usual continuous-to-discrete step). Per sqrt(Hz) is the
    pessimistic reading: as plain RMS the noise would be 10x lower at 100 Hz. Stereolabs support says these figures are
    "a bit high for the gyros" (https://community.stereolabs.com/t/clarification-on-zed-x-imu-parameters-calibration-for-vslam/8364).
    Not published, so not modelled: bias stability, random walk, the chip, and the +-0.5% sensitivity error.
    """

    acc_noise_density: float = 2.3e-3 * G0  # m/s^2 per sqrt(Hz)
    gyro_noise_density: float = float(np.deg2rad(0.20))  # rad/s per sqrt(Hz)
    acc_resolution: float = 0.36e-3 * G0  # m/s^2
    gyro_resolution: float = float(np.deg2rad(0.03))  # rad/s
    acc_range: float = 12.0 * G0  # +-, m/s^2 per axis; a reading clips there
    gyro_range: float = float(np.deg2rad(1000.0))  # +-, rad/s per axis
    rate_hz: float = 200.0  # the part's output rate; the sim samples once per physics step (100 Hz at dt 0.01)

    def sigma(self, dt: float) -> tuple[float, float]:
        """(accel, gyro) standard deviation of one sample taken every ``dt`` seconds."""
        root_hz = float(np.sqrt(1.0 / dt))
        return self.acc_noise_density * root_hz, self.gyro_noise_density * root_hz


@dataclass(frozen=True)
class JointTorqueSpec:
    """Joint torque, with the noise of an estimate from the actuator's motor current (the MH3 arm publishes no joint
    torque sensor). What Genesis's JointTorque returns is the joint's output side (control torque minus the rotor's
    armature times acceleration, plus damping), the quantity a link-side torque sensor measures; a real current-based
    estimate also carries the rotor's inertia and the gearbox friction, so this reading is cleaner than one would be
    except for the noise, which is the current estimate's (below). No actuator vendor
    publishes a figure for a current-based estimate (Unitree, CubeMars, MAB, Robstride, MyActuator, Kinova checked);
    the number is a bench measurement on a quasi-direct-drive motor, 0 to 2 N*m loads: motor-current estimate RMSE
    0.1638 N*m (R^2 0.9609). Grinberg & Ding, arXiv 2510.10843, Fig. 7: https://arxiv.org/pdf/2510.10843v1
    The RMSE is used as white noise: pessimistic for jitter, and it leaves out the estimate's systematic friction error.
    (A link-side torque sensor would be quieter: Franka Panda datasheet, torque noise RMS < 0.055 N*m, resolution
    < 0.02 N*m, https://centers.ulbsibiu.ro/incon/wp-content/uploads/Datasheet-Franka-Emika-Panda-Research-collaborative-robot.pdf
    but those rows most likely describe the estimated end-effector torque, not one joint, so they are not used.)
    """

    noise: float = 0.1638  # N*m, one standard deviation per sample
    resolution: float = 0.0  # no figure published for a current-based estimate


@dataclass(frozen=True)
class SkinSpec:
    """Tactile skin on the palm and phalanges: Tekscan FlexiForce A201, the 4.4 N (1 lb) range. Source, read Oct 21:
    https://www.tekscan.com/products-solutions/force-sensors/a201
    Repeatability < +-2.5% (at 80% of full force), used as one standard deviation of noise proportional to the force,
    so an untouched link reads 0 as the resistive part does (additive noise of the same size made every link read about
    0.15 N untouched, Oct 21 CPU check). The reading clips at the 4.4 N range on its magnitude (one element senses one
    force, so no reading may exceed the range; clipping each axis let a slanted push read up to 4.4 * sqrt(3) = 7.6 N).
    The xyz direction is a sim convenience: a real A201 reports only the force normal to it. Resolution is not listed
    (analog), so none. Linearity < +-3% FS, hysteresis < 4.5% FS and drift are not modelled. (The capacitive PPS
    DigiTacts pad was the first choice for a skin, but its sheet gives pressure only and no element area, so it cannot
    be turned into newtons.)
    """

    repeatability: float = 0.025  # one standard deviation / force, per reading
    max_force: float = 4.4  # N, on the force's magnitude

    def reading(self, true_xyz: np.ndarray, z_xyz: np.ndarray | None = None) -> np.ndarray:
        """What the skin reports for a true force xyz: ``z_xyz`` standard deviations of noise per axis, clipped."""
        force = np.asarray(true_xyz, dtype=float)
        if z_xyz is not None:
            force = force * (1.0 + self.repeatability * np.asarray(z_xyz, dtype=float))
        magnitude = float(np.linalg.norm(force))
        return force * (self.max_force / magnitude) if magnitude > self.max_force else force


@dataclass(frozen=True)
class FingertipSpec:
    """A fingertip force sensor: Interlink FSR 400, datasheet PDS-10004-C:
    https://www.interlinkelectronics.com/downloads/datasheets/fsr-400-series-datasheet.pdf
    Actuation force ~0.2 N min (below it the sensor reads nothing), range ~0.2 to 20 N, resolution continuous (analog),
    repeatability +-2% single part and +-6% part to part, all "(one standard deviation/mean)" measured at 1 kg. So each
    reading carries 2% noise proportional to the force, and each sensor a gain fixed for the run, drawn once with 6%
    spread. Not modelled: the +10% hysteresis and the < 5% per log10(time) drift. An FSR senses force normal to its pad;
    the USD does not say which link axis the pad faces, so the magnitude of the contact force stands in (with friction
    mu it reads at most sqrt(1 + mu^2) high).
    """

    min_force: float = 0.2  # N
    max_force: float = 20.0  # N
    repeatability: float = 0.02  # one standard deviation / mean, per reading
    part_to_part: float = 0.06  # one standard deviation / mean, per sensor

    def reading(self, true_n: float, gain: float = 1.0, z: float = 0.0) -> float:
        """What the sensor reports for a true force: gained, with ``z`` standard deviations of repeatability noise,
        zero below the actuation force, clipped at the range."""
        measured = max(0.0, float(true_n) * gain * (1.0 + self.repeatability * z))
        return 0.0 if measured < self.min_force else min(measured, self.max_force)


DEFAULT_SENSORS = Path(__file__).resolve().parents[3] / "config" / "sensors.yaml"


@dataclass(frozen=True)
class SensorConfig:
    """Which sensors the robot carries and where (``config/sensors.yaml``). Off means every ``read_*`` returns empty."""

    enabled: bool = True
    imus: tuple[tuple[str, str], ...] = (
        ("left_hand", "left_gripper_base"),
        ("head", "head_camera_mount"),
        ("base", "mobile_base"),
        ("torso", "torso_base"),
    )  # (placement, kit link name)
    torque_arms: tuple[str, ...] = ("left",)
    tactile_arms: tuple[str, ...] = ("left",)
    fingertip_arms: tuple[str, ...] = ("left",)
    seed: int = 0
    imu: ImuSpec = field(default_factory=ImuSpec)
    torque: JointTorqueSpec = field(default_factory=JointTorqueSpec)
    skin: SkinSpec = field(default_factory=SkinSpec)
    fingertip: FingertipSpec = field(default_factory=FingertipSpec)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SENSORS) -> SensorConfig:
        """Placements, arms and the on/off flag from YAML; the part specs stay the cited defaults above."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - {"enabled", "seed", "imu_placements", "joint_torque_arms", "tactile_arms", "fingertip_arms"}
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        base = cls()
        placements = raw.get("imu_placements")
        return cls(
            enabled=bool(raw.get("enabled", base.enabled)),
            seed=int(raw.get("seed", base.seed)),
            imus=base.imus if placements is None else tuple((str(k), str(v)) for k, v in placements.items()),
            torque_arms=tuple(raw.get("joint_torque_arms", base.torque_arms)),
            tactile_arms=tuple(raw.get("tactile_arms", base.tactile_arms)),
            fingertip_arms=tuple(raw.get("fingertip_arms", base.fingertip_arms)),
        )

    @classmethod
    def off(cls) -> SensorConfig:
        return cls(enabled=False)

    def arms(self, kind: str) -> tuple[str, ...]:
        """The arms that carry this kind of sensor ("torque", "tactile", "fingertip"); none when the sensors are off."""
        if not self.enabled:
            return ()
        return {"torque": self.torque_arms, "tactile": self.tactile_arms, "fingertip": self.fingertip_arms}[kind]

    @property
    def placements(self) -> tuple[tuple[str, str], ...]:
        return self.imus if self.enabled else ()


@dataclass(frozen=True)
class Pose:
    """World-frame position (m) and orientation quaternion (w, x, y, z)."""

    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class ObjectState:
    """Pose and linear velocity (m/s) of a free object."""

    pos: np.ndarray
    quat: np.ndarray
    lin_vel: np.ndarray


@dataclass(frozen=True)
class Frame:
    """One camera capture: RGB (H, W, 3) uint8 and optionally metric depth (H, W) float."""

    rgb: np.ndarray | None
    depth: np.ndarray | None


@dataclass(frozen=True)
class BoxSpec:
    """A box in the scene. ``fixed`` boxes never move (table, bin walls); free boxes are parts."""

    name: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    yaw_deg: float = 0.0
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    fixed: bool = False
    rho: float = 400.0
    friction: float = 1.0
    texture: np.ndarray | None = field(default=None, compare=False)  # RGB uint8 image wrapped on every face
    finish: str | None = None  # the true finish label, known to the sim only
    # "box"; "capsule": a pill lying along its own x, size = (length, width, height), height <= width flattens it; "disc": a flat round,
    # size = (diameter, diameter, thickness). Every shape's size is its bounding box, so footprints stay boxes.
    shape: Literal["box", "capsule", "disc"] = "box"
    look: Literal["plastic", "matte", "glossy"] = "plastic"  # matte: parchment; glossy: a softgel or a wet pool
    collision: bool = True  # False: drawn only (a pool of liquid); physics and support checks ignore it


@dataclass(frozen=True)
class CameraSpec:
    """A fixed camera in the world (not on the robot), rendered on request by name."""

    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float = 35.0
    res: tuple[int, int] = (480, 300)


@dataclass(frozen=True)
class SceneSpec:
    """Everything the simulator adds to the kit's floor-plus-robot world."""

    boxes: tuple[BoxSpec, ...] = field(default_factory=tuple)
    robot_start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw_deg
    cameras: tuple[CameraSpec, ...] = field(default_factory=tuple)
    lamps: tuple[tuple[float, float, float, float], ...] = ()  # point lights: x, y, z, intensity (empty: the kit's one sun)
    ambient: float | None = None  # ambient light level, 0 to 1 (None: the kit's 0.1)
    sun: float = 5.0  # the kit's directional sun's intensity when lamps are set (0: no sun; probe_tray_light.py, Nov 6)
    kit_floor: bool = True  # False: the kit's checker plane is not drawn (it still collides); the scene's boxes draw the floor

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(box.name for box in self.boxes)

    @property
    def free_names(self) -> tuple[str, ...]:
        return tuple(box.name for box in self.boxes if not box.fixed)

    def box(self, name: str) -> BoxSpec:
        for box in self.boxes:
            if box.name == name:
                return box
        raise KeyError(name)


class Simulator(ABC):
    """What a simulator must provide for the skills, checks and tasks to run on it.

    Single environment. Nothing here assumes a viewer or wall-clock pacing, so a batched
    implementation (many copies of the scene on the GPU, for the RL stage) can offer the same
    methods vectorised. The one per-step Python hook is ``add_controller`` (the grip keeper);
    a batched implementation would run that reflex as a vectorised step too.
    """

    #: The fixed box the last ``move_wrist`` stopped on (see ``stop_on``; estimated from the hand's position), else None.
    last_bump: str | None = None
    #: The sensors this robot carries (``config/sensors.yaml``); the ``read_*`` methods return empty for any left out.
    sensors: SensorConfig = SensorConfig()

    # ----- world -----
    @property
    @abstractmethod
    def dt(self) -> float:
        """Physics step length in simulated seconds."""

    @property
    @abstractmethod
    def time(self) -> float:
        """Simulated seconds since the last reset."""

    @abstractmethod
    def build(self, scene: SceneSpec) -> None:
        """Create the world once. Expensive in Genesis (kernel compile), so build once, reset often."""

    @abstractmethod
    def reset(self) -> None:
        """Return everything to the built state: objects, robot joints, base pose, no welds."""

    @abstractmethod
    def step(self, n: int = 1) -> None:
        """Advance physics by ``n`` steps. Runs every installed controller (``add_controller``) once per step."""

    # ----- per-step controllers: a reflex the robot runs on its own while the skills wait (the grip keeper) -----
    def add_controller(self, key: str, fn: Callable[[], None]) -> None:
        """Run ``fn()`` after every physics step until ``remove_controller(key)`` (or ``reset``) takes it off; a second
        controller under the same key replaces the first. A controller reads sensors and sets joint targets; it must
        not step the sim itself."""
        self.__dict__.setdefault("_controllers", {})[key] = fn

    def remove_controller(self, key: str) -> Callable[[], None] | None:
        return self.__dict__.get("_controllers", {}).pop(key, None)

    def controller(self, key: str) -> Callable[[], None] | None:
        return self.__dict__.get("_controllers", {}).get(key)

    def clear_controllers(self) -> None:
        self.__dict__["_controllers"] = {}

    def _run_controllers(self) -> None:
        for fn in list(self.__dict__.get("_controllers", {}).values()):
            fn()

    def settle(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.dt))))

    # ----- objects -----
    @abstractmethod
    def object_state(self, name: str) -> ObjectState: ...

    @abstractmethod
    def set_object_pose(self, name: str, pos: tuple[float, float, float], yaw_deg: float = 0.0) -> None:
        """Teleport a free object and zero its velocity."""

    # ----- arms and hands -----
    @abstractmethod
    def wrist_pose(self, arm: Arm) -> Pose: ...

    @abstractmethod
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
        """Solve IK for the wrist pose, move the arm there smoothly, wait until within ``tol_m``.

        The wrist follows the straight line to ``pos`` over ``duration_s`` (chosen from the distance when None),
        eased, so the arm does not whip and drop what it holds. Returns the final position error in metres, for
        the caller to judge.

        ``free_yaw``: hold the hand's tilt from ``quat`` (level, a held part upright) and let it turn about world
        vertical. The crossings use it: holding the whole orientation along a line drives the MH3's wrist joints
        (17 degrees each way) into their stops, and the arm's IK then flips pose mid-move (Oct 21,
        scripts/probe_crossing_trace.py). A line whose IK still jumps between points falls back to the joint path to
        the goal: the goal solved fresh from the start pose, or the line's own answer when that one is nearer the start
        joints. Each fallback is recorded in ``line_fallback_log`` (the demo's run log shows it).

        ``stop_on`` names fixed boxes (table, tray pieces) the hand must not touch: every step the
        move feeds the robot's own bump detector (``handoff.bump``, ``config/bump.yaml``: an IMU's
        accelerometer, the joint-torque observer or the tactile skin); when it fires the arm holds
        where it is, ``last_bump`` names the guarded box nearest the hand in the robot's picture of
        the station (an estimate: no detector knows which box it hit; ``last_bump_info`` has the
        detector's numbers) and the move returns. Every call resets ``last_bump`` to None first.
        """

    @abstractmethod
    def set_hand(self, arm: Arm, closed: float) -> None:
        """Position-control the fingers and thumb: 0 fully open, 1 fully closed."""

    def set_thumb(self, arm: Arm, closed: float) -> bool:
        """Position-control the thumb alone (0 open, 1 closed), leaving the fingers where they are commanded; call it
        after ``set_hand``, which sets the thumb too. False when this hand cannot move its thumb on its own."""
        return False

    def thumb_closure(self, arm: Arm) -> float:
        """Measured closure of the thumb, 0 open to 1 closed (the fingers' closure where the thumb is not separate)."""
        return self.hand_closure(arm)

    @abstractmethod
    def squeeze(self, arm: Arm, effort: float) -> None:
        """Force-control the four finger roots with a constant torque (N*m, positive closes).

        Position control commands an opening, so it depends on the part's exact width; a steady
        squeeze closes until the part stops the fingers. ``set_hand`` switches back.
        """

    @abstractmethod
    def hand_closure(self, arm: Arm) -> float:
        """Measured closure of the fingers, 0 open to 1 closed."""

    @abstractmethod
    def hand_contacts(self, arm: Arm, object_name: str) -> int:
        """Number of contact points between any link of that hand and the object, after the last step.

        TRUTH ONLY: the contact solver's answer, which no real robot has (``truth_hand_contacts`` is the same query
        under its honest name). The robot's own touch is ``read_tactile`` and ``read_fingertips``."""

    @abstractmethod
    def attach(self, object_name: str, arm: Arm, link: str | None = None) -> None:
        """Weld the object to the wrist (or to the named robot link, e.g. a fingertip) at their current relative pose."""

    @abstractmethod
    def detach(self, object_name: str) -> None: ...

    def brake(self, on: bool, anchor: str | None = None) -> None:
        """Hold the mobile base where it stands (wheels locked), anchored to the fixed box ``anchor``; ``on=False``
        lets it drive again. A parked robot in a cell brakes (dup, Nov 5). Default: no brake (the fake never slides)."""

    # ----- joints, head, torso -----
    @abstractmethod
    def set_joints(self, targets: Mapping[str, float], *, immediate: bool = False) -> None:
        """Command joint position targets by the kit's public names (teleport when ``immediate``)."""

    @abstractmethod
    def joint_positions(self, names: tuple[str, ...] | None = None) -> dict[str, float]: ...

    @abstractmethod
    def set_head(self, yaw: float, pitch: float) -> None: ...

    # ----- base -----
    @abstractmethod
    def drive(self, linear_mps: float, angular_radps: float) -> None: ...

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    @abstractmethod
    def base_pose(self) -> Pose:
        """World pose of the mobile base link."""

    # ----- sensing -----
    @abstractmethod
    def capture(self, camera: str = "head") -> Frame: ...

    @abstractmethod
    def capture_cloud(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """A fixed camera's colour image (H, W, 3) uint8 and the world position each pixel sees (H, W, 3)."""

    @abstractmethod
    def tray_load(self, boxes: tuple[str, ...]) -> float:
        """Downward force (N) that free objects press on these fixed boxes (one tray's floor and walls), after the
        last step: what a load cell under the tray reads, before its own noise (``handoff.sensors`` adds that)."""

    # ----- the robot's own sensors (SensorConfig): readings as the real parts report them, after the last step -----
    #
    # Every read is a view of what the last physics step measured: none of them steps the sim, so all are cheap enough
    # for a step hook, and reading twice in one step gives the same values.

    @abstractmethod
    def read_imus(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Every IMU by placement (``left_hand``, ``head``, ``base``, ``torso``): (specific force xyz m/s^2, angular
        rate xyz rad/s) in the sensor's own frame, the link's. At rest the accelerometer reads +9.81 m/s^2 up, as a
        real one does. ZED X IMU noise and resolution, clipped at its range (``ImuSpec``)."""

    @abstractmethod
    def read_joint_torques(self, arm: Arm) -> np.ndarray:
        """Torque (N*m) at that arm's 7 joints, ``<arm>_1`` to ``<arm>_7``, as the actuators estimate it from motor
        current: gravity, motion and contact loads included (``JointTorqueSpec``). Empty when that arm has none."""

    @abstractmethod
    def read_tactile(self, arm: Arm) -> dict[str, np.ndarray]:
        """The tactile skin of that hand: contact force xyz (N, link frame) on each of ``skin_links(arm)``, with the
        skin's noise and range (``SkinSpec``). Empty when that hand has none."""

    @abstractmethod
    def read_fingertips(self, arm: Arm) -> dict[str, float]:
        """Each fingertip sensor of that hand by tip link name (``left_index_tip`` ...): force (N) as the FSR reports
        it, 0 below its 0.2 N actuation force (``FingertipSpec``). Empty when that hand has none."""

    @abstractmethod
    def truth_hand_contacts(self, arm: Arm, boxes: tuple[str, ...]) -> dict[str, int]:
        """TRUTH ONLY, never a sensor: contact points between any link of that hand and each named box, after the last
        step, from the simulator's contact solver (the query ``hand_contacts`` uses; the ``stop_on`` guard only in its
        ``contact_truth`` reference mode, never by default)."""

    @abstractmethod
    def truth_finger_penetration(self, arm: Arm, part: str) -> dict[str, float]:
        """TRUTH ONLY, never a sensor: how deep (m) each link of that hand sinks into the part, deepest contact per
        link, after the last step; links not touching it are left out. What a grasp check can score a squeeze by."""

    def sensor_windows(self) -> dict[str, list[dict[str, str]]]:
        """The panel's sensor windows for every sensor carried, as ``RunStatus.set_sensor`` rows: name -> rows.

        Reads only (no stepping); a window per kind, so a person sees each sensor the robot has."""
        windows: dict[str, list[dict[str, str]]] = {}
        imus = self.read_imus()
        if imus:
            acc_max, gyro_max = self.sensors.imu.acc_range, self.sensors.imu.gyro_range
            rows = []
            for placement, (acc, gyro) in imus.items():
                saturated = bool(np.any(np.abs(acc) >= acc_max) or np.any(np.abs(gyro) >= gyro_max))
                rows.append({
                    "label": placement,
                    "value": f"{float(np.linalg.norm(acc)):.2f} m/s^2",
                    "detail": f"gyro {float(np.linalg.norm(gyro)):.3f} rad/s" + (", at its range" if saturated else ""),
                    "state": "warn" if saturated else "ok",
                })
            windows["IMUs (ZED X IMU)"] = rows
        for arm in self.sensors.arms("torque"):
            torques = self.read_joint_torques(arm)  # type: ignore[arg-type]
            windows[f"{arm.capitalize()} arm joint torque"] = [
                {"label": f"{arm}_{i}", "value": f"{float(t):+.2f} N*m", "detail": "joint output torque, current-estimate noise", "state": "ok"}
                for i, t in enumerate(torques, start=1)
            ]
        for arm in self.sensors.arms("tactile"):
            skin_max = self.sensors.skin.max_force
            rows = []
            for link, force in self.read_tactile(arm).items():  # type: ignore[arg-type]
                magnitude = float(np.linalg.norm(force))
                full = magnitude >= skin_max - 1e-9
                rows.append({
                    "label": link.removeprefix(f"{arm}_"),
                    "value": f"{magnitude:.2f} N",
                    "detail": "at its 4.4 N range" if full else ("touching" if magnitude > 0 else "no touch"),
                    "state": "warn" if full else ("ok" if magnitude > 0 else "idle"),
                })
            windows[f"{arm.capitalize()} hand tactile skin"] = rows
        for arm in self.sensors.arms("fingertip"):
            tip_max = self.sensors.fingertip.max_force
            windows[f"{arm.capitalize()} fingertips"] = [
                {
                    "label": tip.removeprefix(f"{arm}_"),
                    "value": f"{force:.2f} N",
                    "detail": "at its 20 N range" if force >= tip_max else ("pressing" if force > 0 else "below 0.2 N"),
                    "state": "warn" if force >= tip_max else ("ok" if force > 0 else "idle"),
                }
                for tip, force in self.read_fingertips(arm).items()  # type: ignore[arg-type]
            ]
        return windows


def yaw_of(quat_wxyz: np.ndarray) -> float:
    """Heading (rad) about world z of a quaternion, for base and part orientation checks."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_from_yaw(yaw_rad: float) -> np.ndarray:
    return np.array([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of two (w, x, y, z) quaternions: rotate by b, then by a."""
    w1, x1, y1, z1 = (float(v) for v in a)
    w2, x2, y2, z2 = (float(v) for v in b)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def line_fallback_line(info: Mapping[str, Any]) -> str:
    """The run-log line for one straight-line move that fell back to the joint path (GenesisSimulator.move_wrist)."""
    goal = "the goal solved fresh from the start pose" if info["goal"] == "fresh" else "the chained solution (the fresh one swung further)"
    target = ", ".join(f"{float(v):.2f}" for v in info["target"])
    return (
        f"straight line given up at t {float(info['t']):.2f} s ({info['arm']} wrist to {target}): the IK jumped "
        f"{float(info['jump_rad']):.2f} rad between 1 cm points, so the arm took the joint path to {goal}, "
        f"up to {float(info['swing_rad']):.2f} rad per joint"
    )
