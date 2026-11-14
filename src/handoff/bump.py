"""Bump detection from the robot's own sensors: did the hand (or what it carries) just hit something fixed?

A guarded wrist move (``Simulator.move_wrist(stop_on=...)``) runs one of these detectors every physics step and stops
the arm when it fires; the stop, the report and the retry are the skills' (``skills.travel`` / ``back_off``). Which
detector decides is ``config/bump.yaml``'s ``bump_sensor``:

- ``hand_imu`` / ``head_imu`` / ``base_imu`` / ``torso_imu``: an IMU placement's accelerometer, high-passed. The
  residual is the measured specific force minus its own mean over the previous ``baseline_s``: gravity, the hand's
  slow rotation and the eased commanded motion (under about 4 m/s^2, changing over the whole 0.3 to 2 s of the move,
  and smoothed again by the joint PD) sit in the baseline; an impact, which stops the hand within one or two 10 ms
  steps, does not. Not the commanded trajectory's second derivative: the arm lags its command by about kv/kp (0.1 s on
  the MH3 gains), so the command predicts the actual acceleration badly exactly when it changes. A slow press makes no
  impact, so an IMU is expected to miss it (the torque detector is the one for that).
- ``joint_torque``: a momentum-observer residual on the arm's 7 joints (De Luca and Mattone, "Sensorless robot
  collision detection and hybrid force/motion control", ICRA 2005; Haddadin et al., "Robot collisions: a survey on
  detection, isolation, and identification", IEEE T-RO 33(6) 2017). The external torque is what the measured joint
  torque and the robot's dynamics model (mass matrix, gravity and Coriolis) do not explain:
  ``tau_ext = M dqdot/dt + bias - tau``, and the observer is that residual through a first-order filter of gain
  ``gain`` (1/s), which is what the integral momentum form computes without differentiating velocity. It is anchored to
  the model, not to the signal's own past, so a press that builds slowly still shows. Chosen over a per-joint baseline
  from the commanded trajectory because the joint PD's effort IS the measured torque, so a baseline from the command
  only re-derives the reading, and gravity torque changes by several N*m across a crossing, which a fixed baseline
  would call a bump. The residual is tared over the move's first ``tare_s`` (the carried part's weight and the model's
  constant error); a bump inside that window is missed.
- ``tactile``: the hand's skin (``read_tactile``), the largest change of any link's force since the move started
  (tared over ``tare_s``, because a carried part loads the palm and phalanges all along).
- ``contact_truth``: the simulator's contact solver. TRUTH ONLY, the reference the others are scored against; never
  the default, and a bump it reports is the true box, not an estimate.

No detector knows WHICH box it hit. ``estimate_box`` names the fixed box nearest the hand as the robot believes the
station to be: the drawing (the scene as built), with each tray moved to where the bins camera saw it (``bins_seen``).
Every report says it is an estimate.

The detectors are streaming and pure (numpy in, numbers out): the simulator feeds them online, and
``scripts/score_bump_sensors.py`` feeds the same classes the recorded probe traces, so a threshold picked offline
behaves the same online.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BUMP = Path(__file__).resolve().parents[2] / "config" / "bump.yaml"
IMU_SENSORS = {"hand_imu": "{arm}_hand", "head_imu": "head", "base_imu": "base", "torso_imu": "torso"}
BUMP_SENSORS = (*IMU_SENSORS, "joint_torque", "tactile", "contact_truth")
# Who decides when the configured sensor is not carried on the moving arm: the arm's own sensors first, then the body's.
FALLBACK_ORDER = ("hand_imu", "joint_torque", "tactile", "torso_imu", "head_imu", "base_imu")
UNITS = {"imu": "m/s^2", "torque": "N*m", "tactile": "N", "truth": "contacts"}


# ----- configuration -----


@dataclass(frozen=True)
class ImuDetectorConfig:
    threshold: float = 3.0  # m/s^2 on the residual's norm; above a clean move's peak in the Oct 21 CPU smoke (yaml says why)
    baseline_s: float = 0.05  # the mean the residual is taken against: the previous 5 steps at dt 0.01
    hold: int = 2  # steps the residual must stay over the threshold in a row


@dataclass(frozen=True)
class TorqueDetectorConfig:
    threshold: float = 0.5  # N*m on the largest joint residual; 6 sigma of the observer's noise (yaml says why)
    gain: float = 25.0  # 1/s, the observer's bandwidth: the residual reaches 63% of a step load in 1/gain = 40 ms
    tare_s: float = 0.1  # the residual's mean over this long at the move's start is its zero
    hold: int = 2


@dataclass(frozen=True)
class TactileDetectorConfig:
    threshold: float = 0.5  # N, the largest change of one skin link's force since the tare
    tare_s: float = 0.05
    hold: int = 1


@dataclass(frozen=True)
class BumpConfig:
    """Which detector stops a guarded move (``sensor``), and every detector's settings (``config/bump.yaml``)."""

    sensor: str = "hand_imu"
    imu: ImuDetectorConfig = field(default_factory=ImuDetectorConfig)
    torque: TorqueDetectorConfig = field(default_factory=TorqueDetectorConfig)
    tactile: TactileDetectorConfig = field(default_factory=TactileDetectorConfig)
    monitor_all: bool = True  # also run every other detector the robot carries, for the panel and the log (they never stop the arm)
    truth_every: int = 5  # contact_truth only: steps between contact-solver reads (one read of the whole robot each)

    def __post_init__(self) -> None:
        if self.sensor not in BUMP_SENSORS:
            raise ValueError(f"bump_sensor {self.sensor!r}: one of {BUMP_SENSORS}")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_BUMP) -> BumpConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sections = {"imu": ImuDetectorConfig, "joint_torque": TorqueDetectorConfig, "tactile": TactileDetectorConfig}
        unknown = set(raw) - {"bump_sensor", "monitor_all", "truth_every", *sections}
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        parts: dict[str, Any] = {}
        for key, kind in sections.items():
            given = raw.get(key) or {}
            bad = set(given) - set(kind.__dataclass_fields__)
            if bad:
                raise ValueError(f"unknown keys under {key} in {path}: {sorted(bad)}")
            parts["torque" if key == "joint_torque" else key] = kind(**{k: (int(v) if k == "hold" else float(v)) for k, v in given.items()})
        base = cls()
        return cls(
            sensor=str(raw.get("bump_sensor", base.sensor)),
            monitor_all=bool(raw.get("monitor_all", base.monitor_all)),
            truth_every=int(raw.get("truth_every", base.truth_every)),
            **parts,
        )

    def with_sensor(self, sensor: str) -> BumpConfig:
        return replace(self, sensor=sensor)


# ----- the detectors: streaming, one sample per physics step -----


class StreamDetector:
    """One statistic per step; fires when it stays over ``threshold`` for ``hold`` steps in a row.

    ``fired_at`` is the step index (0 = the move's first step) at which it fired, else None; ``stat`` the latest value,
    ``peak`` the largest so far. The statistic never depends on the threshold, so offline scoring can compute it once
    and try thresholds on it (``first_alarm``)."""

    kind = ""

    def __init__(self, name: str, threshold: float, hold: int, dt: float) -> None:
        self.name, self.threshold, self.hold, self.dt = name, float(threshold), max(1, int(hold)), float(dt)
        self.n = 0
        self.stat = 0.0
        self.peak = 0.0
        self.above = 0
        self.fired_at: int | None = None

    def statistic(self, *sample: Any) -> float:
        raise NotImplementedError

    def update(self, *sample: Any) -> bool:
        """Feed one step's sample; True once the detector has fired (and from then on)."""
        self.stat = float(self.statistic(*sample))
        self.peak = max(self.peak, self.stat)
        self.above = self.above + 1 if self.stat > self.threshold else 0
        if self.fired_at is None and self.above >= self.hold:
            self.fired_at = self.n
        self.n += 1
        return self.fired_at is not None

    @property
    def fired(self) -> bool:
        return self.fired_at is not None

    @property
    def unit(self) -> str:
        return UNITS[self.kind]


class ImuBumpDetector(StreamDetector):
    """Accelerometer residual: specific force minus its mean over the previous ``baseline_s``, as a norm (m/s^2).
    Zero while that baseline is still filling (the move's first ``baseline_s``)."""

    kind = "imu"

    def __init__(self, name: str, cfg: ImuDetectorConfig, dt: float) -> None:
        super().__init__(name, cfg.threshold, cfg.hold, dt)
        self._past: deque[np.ndarray] = deque(maxlen=max(1, int(round(cfg.baseline_s / dt))))

    def statistic(self, acc: np.ndarray) -> float:
        acc = np.asarray(acc, dtype=float).reshape(3)
        full = len(self._past) == self._past.maxlen
        value = float(np.linalg.norm(acc - np.mean(self._past, axis=0))) if full else 0.0
        self._past.append(acc)
        return value


class TorqueBumpDetector(StreamDetector):
    """Momentum-observer residual (N*m per joint); the statistic is the largest joint's magnitude.

    One sample is ``(tau, dp, bias)``: the measured joint torque, the change of generalized momentum over the step
    (``M (qdot_after - qdot_before)``, the arm's rows of the robot's mass matrix) and the gravity and Coriolis torque
    (``bias``), all for the arm's joints. ``tau_ext = dp / dt + bias - tau``; the observer filters it with ``gain``
    (``r += gain * dt * (tau_ext - r)``), after a tare over the first ``tare_s``. A sample of None (the model terms are
    not ready, e.g. the move's first step has no previous velocity) counts as a step of zero."""

    kind = "torque"

    def __init__(self, name: str, cfg: TorqueDetectorConfig, dt: float) -> None:
        super().__init__(name, cfg.threshold, cfg.hold, dt)
        self.alpha = float(np.clip(cfg.gain * dt, 0.0, 1.0))
        self._tare_n = max(1, int(round(cfg.tare_s / dt)))
        self._tare: list[np.ndarray] = []
        self.offset: np.ndarray | None = None
        self.residual: np.ndarray | None = None  # per joint, N*m, after the tare

    def statistic(self, tau: np.ndarray | None, dp: np.ndarray | None = None, bias: np.ndarray | None = None) -> float:
        if tau is None or dp is None or bias is None:
            return 0.0 if self.residual is None else float(np.max(np.abs(self.residual)))
        external = np.asarray(dp, dtype=float) / self.dt + np.asarray(bias, dtype=float) - np.asarray(tau, dtype=float)
        if self.offset is None:
            self._tare.append(external)
            if len(self._tare) >= self._tare_n:
                self.offset = np.mean(self._tare, axis=0)
                self.residual = np.zeros_like(self.offset)
            return 0.0
        assert self.residual is not None
        self.residual = self.residual + self.alpha * (external - self.offset - self.residual)
        return float(np.max(np.abs(self.residual)))


class TactileBumpDetector(StreamDetector):
    """Skin force change (N): the largest change of any skin link's force magnitude since the tare. Magnitudes, not
    vectors: a real A201 element reports one force, so a change of direction at the same size must not count."""

    kind = "tactile"

    def __init__(self, name: str, cfg: TactileDetectorConfig, dt: float) -> None:
        super().__init__(name, cfg.threshold, cfg.hold, dt)
        self._tare_n = max(1, int(round(cfg.tare_s / dt)))
        self._tare: list[np.ndarray] = []
        self.offset: np.ndarray | None = None

    def statistic(self, forces: np.ndarray) -> float:
        magnitudes = np.linalg.norm(np.asarray(forces, dtype=float).reshape(-1, 3), axis=1)
        if self.offset is None:
            self._tare.append(magnitudes)
            if len(self._tare) >= self._tare_n:
                self.offset = np.mean(self._tare, axis=0)
            return 0.0
        return float(np.max(np.abs(magnitudes - self.offset))) if len(magnitudes) else 0.0


class TruthBumpDetector(StreamDetector):
    """TRUTH ONLY: contact points between the hand and the guarded boxes, from the simulator's contact solver."""

    kind = "truth"

    def __init__(self, name: str, dt: float) -> None:
        super().__init__(name, 0.0, 1, dt)

    def statistic(self, count: float) -> float:
        return float(count)


def make_detector(sensor: str, cfg: BumpConfig, dt: float) -> StreamDetector:
    """A fresh detector for one bump sensor name (``BUMP_SENSORS``)."""
    if sensor in IMU_SENSORS:
        return ImuBumpDetector(sensor, cfg.imu, dt)
    if sensor == "joint_torque":
        return TorqueBumpDetector(sensor, cfg.torque, dt)
    if sensor == "tactile":
        return TactileBumpDetector(sensor, cfg.tactile, dt)
    if sensor == "contact_truth":
        return TruthBumpDetector(sensor, dt)
    raise ValueError(f"no bump sensor {sensor!r}: one of {BUMP_SENSORS}")


# ----- offline: the same detectors over recorded traces -----


def statistics(detector: StreamDetector, samples: Sequence[tuple[Any, ...]]) -> np.ndarray:
    """Feed a fresh detector every step's sample (a tuple of its ``update`` arguments) and return the statistic per step."""
    out = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        detector.update(*sample)
        out[i] = detector.stat
    return out


def held(stats: np.ndarray, hold: int) -> np.ndarray:
    """Per step, the smallest statistic of the last ``hold`` steps (0 before ``hold`` steps): the detector fires at the
    first step where this exceeds its threshold."""
    stats = np.asarray(stats, dtype=float)
    hold = max(1, int(hold))
    if hold == 1:
        return stats.copy()
    out = np.zeros_like(stats)
    for i in range(hold - 1, len(stats)):
        out[i] = float(np.min(stats[i - hold + 1 : i + 1]))
    return out


def first_alarm(stats: np.ndarray, threshold: float, hold: int) -> int | None:
    """The step at which a detector with this threshold and hold fires on these statistics, else None (as ``update``)."""
    over = np.nonzero(held(stats, hold) > threshold)[0]
    return int(over[0]) if len(over) else None


# ----- which box: an estimate from the robot's own picture of the station -----


@dataclass(frozen=True)
class BelievedBox:
    """A fixed box where the robot believes it is: centre (m), half sizes (m), yaw (rad)."""

    name: str
    center: np.ndarray
    half: np.ndarray
    yaw: float = 0.0

    def distance(self, point: np.ndarray) -> float:
        """Distance (m) from a point to this box's surface; 0 inside it."""
        d = np.asarray(point, dtype=float) - self.center
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        return float(np.linalg.norm(np.maximum(np.abs(local) - self.half, 0.0)))


def believed_boxes(scene: Any, names: Sequence[str], bins_seen: Mapping[str, tuple[float, float]] | None = None) -> dict[str, BelievedBox]:
    """The named fixed boxes of a built ``SceneSpec`` (the drawing), each tray's pieces moved by however far the bins
    camera saw that tray from its drawing position (a tray's pieces are ``<bin>_<piece>``, its centre its ``_floor``)."""
    by_name = {box.name: box for box in scene.boxes}
    seen = dict(bins_seen or {})
    out = {}
    for name in names:
        box = by_name[name]
        center = np.array(box.pos, dtype=float)
        tray = name.rsplit("_", 2)[0] if "_wall_" in name else name.removesuffix("_floor")
        if tray in seen and f"{tray}_floor" in by_name:
            drawn = by_name[f"{tray}_floor"].pos
            center[:2] += np.array(seen[tray], dtype=float) - np.array(drawn[:2], dtype=float)
        out[name] = BelievedBox(name, center, np.array(box.size, dtype=float) / 2.0, float(np.radians(box.yaw_deg)))
    return out


def estimate_box(points: np.ndarray, boxes: Mapping[str, BelievedBox]) -> tuple[str | None, float]:
    """The believed box nearest any of the hand's points (link origins from the arm's own kinematics), and how far (m)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    best: tuple[str | None, float] = (None, float("inf"))
    for name, box in boxes.items():
        d = min(box.distance(p) for p in points)
        if d < best[1]:
            best = (name, d)
    return best


# ----- one guarded move: the deciding detector and, for the panel, the others -----


def placement(sensor: str, arm: str) -> str | None:
    """The IMU placement an IMU bump sensor reads (``hand_imu`` -> ``left_hand`` for the left arm), else None."""
    return IMU_SENSORS[sensor].format(arm=arm) if sensor in IMU_SENSORS else None


class BumpGuard:
    """The detectors of one guarded move. ``step()`` reads the sensors once (after a physics step) and feeds them.

    ``sim`` provides the robot's reads: ``read_imus``, ``read_joint_torques(arm)`` with ``torque_model(arm)`` (the
    dynamics model's ``(dp, bias)`` for the arm, or None until ready), ``read_tactile(arm)``, and only for
    ``contact_truth`` ``truth_hand_contacts(arm, boxes)``. The configured sensor must be one the robot carries."""

    def __init__(self, sim: Any, arm: str, boxes: Sequence[str], cfg: BumpConfig) -> None:
        self.sim, self.arm, self.boxes, self.cfg = sim, arm, tuple(boxes), cfg
        self.sensor = cfg.sensor
        self.fallback_from: str | None = None  # the configured sensor, when this arm does not carry it
        available = self.available(sim, arm)
        if self.sensor != "contact_truth" and self.sensor not in available:
            # A sensor the robot carries on its other arm only (hand_imu on the right arm: config/sensors.yaml fits the
            # left hand) decides with the next carried sensor in FALLBACK_ORDER rather than crash mid-run; the report
            # and the log line say so. A sensor the robot carries nowhere is a config error and is refused.
            other = "right" if arm == "left" else "left"
            substitute = next((s for s in FALLBACK_ORDER if s in available), None)
            if self.sensor not in self.available(sim, other) or substitute is None:
                raise ValueError(
                    f"bump_sensor {self.sensor!r} needs a sensor this robot does not carry for the {arm} arm "
                    f"(config/sensors.yaml); it carries: {sorted(available) or 'none'}"
                )
            self.fallback_from, self.sensor = self.sensor, substitute
        names = sorted(available) if cfg.monitor_all else [self.sensor]
        if self.sensor == "contact_truth" and self.sensor not in names:
            names.append(self.sensor)
        self.detectors: dict[str, StreamDetector] = {name: make_detector(name, cfg, sim.dt) for name in names}
        self.active = True
        self.truth_box: str | None = None  # contact_truth only: the box the solver says the hand touches

    @staticmethod
    def available(sim: Any, arm: str) -> set[str]:
        """The bump sensors (``BUMP_SENSORS`` short of contact_truth) this robot carries for that arm."""
        sensors = sim.sensors
        placements = {p for p, _ in sensors.placements}
        out = {name for name in IMU_SENSORS if placement(name, arm) in placements}
        if arm in sensors.arms("torque"):
            out.add("joint_torque")
        if arm in sensors.arms("tactile"):
            out.add("tactile")
        return out

    @property
    def decider(self) -> StreamDetector:
        return self.detectors[self.sensor]

    def step(self) -> bool:
        """Read and feed every detector once; True when the deciding one has fired."""
        dets = self.detectors
        if any(d.kind == "imu" for d in dets.values()):
            imus = self.sim.read_imus()
            for name, det in dets.items():
                if det.kind == "imu":
                    det.update(imus[placement(name, self.arm)][0])
        if "joint_torque" in dets:
            model = self.sim.torque_model(self.arm)
            tau = self.sim.read_joint_torques(self.arm)
            dets["joint_torque"].update(*((tau, *model) if model is not None else (None,)))
        if "tactile" in dets:
            skin = self.sim.read_tactile(self.arm)
            dets["tactile"].update(np.array([skin[k] for k in sorted(skin)]) if skin else np.zeros((0, 3)))
        if "contact_truth" in dets:
            det = dets["contact_truth"]
            if (det.n + 1) % max(1, self.cfg.truth_every) == 0:
                counts = self.sim.truth_hand_contacts(self.arm, self.boxes)
                touched = [b for b in self.boxes if counts.get(b, 0) > 0]
                if touched and self.truth_box is None:
                    self.truth_box = touched[0]
                det.update(sum(counts.values()))
            else:
                det.update(0.0)
        return self.decider.fired

    def report(self, points: np.ndarray, scene: Any, bins_seen: Mapping[str, tuple[float, float]] | None, t: float) -> dict[str, Any]:
        """What the stop says: the deciding detector's numbers and the box, estimated (or the truth for contact_truth)."""
        det = self.decider
        if self.sensor == "contact_truth":
            box, how, dist = self.truth_box, "contact solver (TRUTH, not a robot sensor)", 0.0
        else:
            box, dist = estimate_box(points, believed_boxes(scene, self.boxes, bins_seen))
            how = "nearest box to the hand in the drawing" + (", trays where the bins camera saw them" if bins_seen else "")
        return {
            "box": box, "estimated": self.sensor != "contact_truth", "how": how, "distance_m": round(float(dist), 4),
            "sensor": self.sensor, "stat": round(det.stat, 4), "threshold": det.threshold, "hold": det.hold,
            "unit": det.unit, "step": det.fired_at, "t": round(float(t), 3),
            "others": {n: d.fired_at for n, d in self.detectors.items() if n != self.sensor},
            "fallback_from": self.fallback_from,
        }


def start_guard(sim: Any, arm: str, stop_on: Sequence[str]) -> BumpGuard | None:
    """A guarded move's start, the same in every simulator: the detectors from ``sim.bump_config`` (``config/bump.yaml``
    when None), kept as ``sim.bump_guard`` for the panel. None for an unguarded move."""
    if not stop_on:
        return None
    if getattr(sim, "bump_config", None) is None:
        sim.bump_config = BumpConfig.from_yaml(DEFAULT_BUMP)
    sim.last_bump_info = None
    sim.bump_guard = BumpGuard(sim, arm, stop_on, sim.bump_config)
    return sim.bump_guard


def record_stop(sim: Any, guard: BumpGuard, points: np.ndarray, scene: Any) -> dict[str, Any]:
    """The guard fired: name the box (``guard.report``) in ``sim.last_bump``, keep the report as ``sim.last_bump_info``
    and in ``sim.bump_log``, and end the guard. The caller holds the arm."""
    info = guard.report(points, scene, getattr(sim, "bins_seen", None), sim.time)
    guard.active = False
    sim.last_bump = info["box"]
    sim.last_bump_info = info
    if getattr(sim, "bump_log", None) is None:
        sim.bump_log = []
    sim.bump_log.append(info)
    return info


def bump_line(info: Mapping[str, Any]) -> str:
    """The run-log line for one detected bump."""
    est = "estimated" if info.get("estimated") else "true"
    others = info.get("others") or {}
    also = [n for n, s in others.items() if s is not None]
    return (
        f"bump felt by {info['sensor']}: {info['stat']:.2f} {info['unit']} over its {info['threshold']:.2f} for {info['hold']} "
        f"step{'s' if info['hold'] != 1 else ''} at t {info['t']:.2f} s; box {info['box']} ({est}: {info['how']}, "
        f"{float(info['distance_m']) * 100:.0f} cm from the hand)" + (f"; also fired: {', '.join(sorted(also))}" if also else "")
        + (f"; {info['fallback_from']} is not carried on this arm, so {info['sensor']} decided" if info.get("fallback_from") else "")
    )


# ----- the panel's windows -----


def _row(label: str, det: StreamDetector, deciding: bool, active: bool, value: float, detail: str) -> dict[str, str]:
    over = value > det.threshold
    state = "idle" if not active else ("warn" if over or det.fired else "ok")
    return {
        "label": label,
        "value": f"{value:.2f} {det.unit}",
        "detail": f"{detail}, threshold {det.threshold:.2f}" + ("; decides the stop" if deciding else "") + ("; FIRED" if det.fired else "") + ("" if active else "; not guarding"),
        "state": state,
    }


def accelerometer_rows(guard: BumpGuard | None) -> list[dict[str, str]]:
    """The "Accelerometer" window: each IMU placement's bump residual against its threshold, during the last guarded move."""
    if guard is None:
        return [{"label": "bump guard", "value": "-", "detail": "no guarded move yet", "state": "idle"}]
    rows = [
        _row(name, det, name == guard.sensor, guard.active, det.stat, "high-passed specific force")
        for name, det in guard.detectors.items()
        if det.kind == "imu"
    ]
    return rows or [{"label": "bump guard", "value": "-", "detail": "no IMU carried", "state": "idle"}]


def torque_rows(guard: BumpGuard | None) -> list[dict[str, str]]:
    """The "Joint torque" window: the momentum-observer residual per arm joint, during the last guarded move."""
    det = None if guard is None else guard.detectors.get("joint_torque")
    if det is None:
        return [{"label": "bump guard", "value": "-", "detail": "no guarded move yet" if guard is None else "no joint torque carried", "state": "idle"}]
    assert isinstance(det, TorqueBumpDetector)
    if det.residual is None:
        return [{"label": f"{guard.arm}_1..7", "value": "taring", "detail": "residual zeroed over the move's start", "state": "idle" if not guard.active else "ok"}]
    return [
        _row(f"{guard.arm}_{i}", det, guard.sensor == "joint_torque", guard.active, abs(float(r)), "external torque (momentum observer)")
        for i, r in enumerate(det.residual, start=1)
    ]


def tactile_rows(guard: BumpGuard | None) -> list[dict[str, str]]:
    """The "Skin (bump)" window: the tactile detector's force change against its threshold, during the last guarded move."""
    det = None if guard is None else guard.detectors.get("tactile")
    if det is None:
        return [{"label": "bump guard", "value": "-", "detail": "no guarded move yet" if guard is None else "no tactile skin carried", "state": "idle"}]
    return [_row("skin", det, guard.sensor == "tactile", guard.active, det.stat, "largest change of any skin link's force")]


__all__ = [
    "BUMP_SENSORS", "DEFAULT_BUMP", "FALLBACK_ORDER", "BelievedBox", "BumpConfig", "BumpGuard", "ImuBumpDetector", "StreamDetector",
    "TactileBumpDetector", "TorqueBumpDetector", "TruthBumpDetector", "accelerometer_rows", "believed_boxes", "bump_line",
    "estimate_box", "first_alarm", "held", "make_detector", "placement", "record_stop", "start_guard", "statistics",
    "tactile_rows", "torque_rows",
]
