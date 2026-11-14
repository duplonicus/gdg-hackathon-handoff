"""Bump detection from the robot's own sensors (handoff.bump): the detectors on synthetic traces, the box estimate, the
fake's guarded moves deciding by the configured detector, and the probe scorer on a tiny synthetic dataset."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from handoff import bump
from handoff.bump import (
    BumpConfig,
    BumpGuard,
    ImuBumpDetector,
    ImuDetectorConfig,
    TactileBumpDetector,
    TactileDetectorConfig,
    TorqueBumpDetector,
    TorqueDetectorConfig,
    believed_boxes,
    estimate_box,
    first_alarm,
    held,
    statistics,
)
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.base import G0, BoxSpec, ImuSpec, JointTorqueSpec, SceneSpec, SensorConfig
from handoff.sim.fake import HOME_WRIST, FakeSimulator
from handoff.skills import HOME_QUAT, TRAVEL_POSE

DT = 0.01
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("score_bump_sensors", SCRIPTS / "score_bump_sensors.py")
score_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_mod)


# ----- synthetic traces -----


def eased(n_move: int, distance: float) -> np.ndarray:
    """The commanded wrist acceleration (m/s^2) of move_wrist's cosine ease over n_move steps, along one axis."""
    t = np.arange(n_move) * DT
    T = n_move * DT
    return distance * (np.pi / T) ** 2 / 2.0 * np.cos(np.pi * t / T)


def lagged(a: np.ndarray, tau_s: float = 0.1) -> np.ndarray:
    """What the arm does with a commanded acceleration: the joint PD follows it with about kv/kp of lag."""
    out = np.zeros_like(a)
    k = DT / tau_s
    for i in range(1, len(a)):
        out[i] = out[i - 1] + k * (a[i] - out[i - 1])
    return out


def imu_trace(n: int, motion: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Specific force (n, 3) of a hand at rest or moving along x, with the ZED X IMU's white noise."""
    acc_sigma, _ = ImuSpec().sigma(DT)
    rng = np.random.default_rng(seed)
    acc = np.tile([0.0, 0.0, G0], (n, 1)) + rng.normal(0.0, acc_sigma, size=(n, 3))
    if motion is not None:
        acc[: len(motion), 0] += motion
    return acc


def torque_trace(n: int, external: np.ndarray, motion: bool = False, seed: int = 0):
    """(tau, dp, bias) per step for a 7-joint arm, with the cited motor-current noise: gravity torque held by the
    motors, optionally an eased joint move (perfectly tracked), and ``external`` (n, 7) pressing on it."""
    rng = np.random.default_rng(seed)
    mass = np.diag([2.0, 1.8, 1.2, 0.9, 0.3, 0.2, 0.1])
    q = np.zeros((n + 1, 7))
    if motion:
        s = 0.5 - 0.5 * np.cos(np.pi * np.clip(np.arange(n + 1) / n, 0, 1))
        q = np.outer(s, [0.6, -0.4, 0.3, 0.8, 0.2, -0.3, 0.1])
    v = np.gradient(q, DT, axis=0)
    out = []
    for k in range(n):
        gravity = np.array([12.0, -8.0, 3.0, 5.0, 0.5, 0.2, 0.1]) * np.cos(q[k])  # changes with the configuration
        dp = mass @ (v[k + 1] - v[k])
        motor = dp / DT + gravity - external[k]  # what the actuators deliver: dynamics minus the external load
        out.append((motor + rng.normal(0.0, JointTorqueSpec().noise, 7), dp, gravity))
    return out


def run(detector, samples):
    for sample in samples:
        detector.update(*sample)
    return detector


# ----- the detectors -----


def test_an_impact_spike_is_caught_by_the_imu_within_two_steps():
    acc = imu_trace(200)
    acc[120:122, 0] -= 20.0  # the hand stops dead against a wall: two 10 ms samples of about 2 g against the motion
    det = run(ImuBumpDetector("hand_imu", ImuDetectorConfig(), DT), [(a,) for a in acc])
    assert det.fired_at == 121  # hold 2: the second spike sample
    assert (det.fired_at - 120) * DT * 1000 <= 20.0


def test_a_smooth_eased_move_is_flagged_by_neither_the_imu_nor_the_torque():
    motion = lagged(eased(120, 0.30))  # 30 cm in 1.2 s, as move_wrist eases it, as the PD follows it
    imu = run(ImuBumpDetector("hand_imu", ImuDetectorConfig(), DT), [(a,) for a in imu_trace(200, motion)])
    assert not imu.fired, imu.peak
    torque = run(TorqueBumpDetector("joint_torque", TorqueDetectorConfig(), DT), torque_trace(200, np.zeros((200, 7)), motion=True))
    assert not torque.fired, torque.peak


def test_the_cited_noise_alone_stays_under_the_default_thresholds_for_a_hundred_moves():
    imu_peaks, torque_peaks = [], []
    for seed in range(100):
        imu_peaks.append(run(ImuBumpDetector("hand_imu", ImuDetectorConfig(), DT), [(a,) for a in imu_trace(150, seed=seed)]).peak)
        torque_peaks.append(run(TorqueBumpDetector("joint_torque", TorqueDetectorConfig(), DT), torque_trace(150, np.zeros((150, 7)), seed=seed)).peak)
    assert max(imu_peaks) < ImuDetectorConfig().threshold
    assert max(torque_peaks) < TorqueDetectorConfig().threshold


def test_a_slow_press_is_caught_by_the_torque_and_not_by_the_imu():
    n = 300
    external = np.zeros((n, 7))
    external[100:, 1] = np.minimum(np.arange(n - 100) * DT * 3.0, 3.0)  # 3 N*m building over a second on the shoulder
    torque = run(TorqueBumpDetector("joint_torque", TorqueDetectorConfig(), DT), torque_trace(n, external))
    assert torque.fired and 100 < torque.fired_at < 140, torque.fired_at  # within 0.4 s of the press starting
    acc = imu_trace(n)
    acc[100:, 0] -= np.minimum(np.arange(n - 100) * DT * 0.05, 0.05)  # the hand slows by a few cm/s^2: no impact
    imu = run(ImuBumpDetector("hand_imu", ImuDetectorConfig(), DT), [(a,) for a in acc])
    assert not imu.fired, imu.peak


def test_the_torque_residual_is_anchored_to_the_model_not_to_its_own_past():
    """A slow ramp that a self-adapting baseline would absorb keeps growing in the observer, and a constant load
    present from the move's start (the carried part's weight) is tared out."""
    n = 400
    external = np.full((n, 7), 0.3)  # the part's weight, there from the start
    external[100:, 3] += np.arange(n - 100) * DT * 0.5  # 0.5 N*m per second: very slow
    det = TorqueBumpDetector("joint_torque", TorqueDetectorConfig(), DT)
    stats = statistics(det, torque_trace(n, external))
    assert stats[90] < 0.3  # the constant 0.3 N*m load is gone after the tare
    assert stats[-1] > 1.2 and det.fired  # 1.5 N*m of slow press shows almost fully


def test_tactile_tares_the_carried_part_and_fires_on_a_new_force():
    grip = np.zeros((11, 3))
    grip[0] = [0.0, 1.2, 0.0]  # the part pressing on the palm all along
    frames = [grip.copy() for _ in range(100)]
    for f in frames[60:]:
        f[3] = [0.0, 0.0, 0.8]  # a wall on a finger segment
    det = run(TactileBumpDetector("tactile", TactileDetectorConfig(), DT), [(f,) for f in frames])
    assert det.fired_at == 60


def test_offline_first_alarm_matches_the_online_detector():
    acc = imu_trace(300, lagged(eased(150, 0.4)), seed=3)
    acc[200:203, 1] += 6.0
    stats = statistics(ImuBumpDetector("hand_imu", ImuDetectorConfig(), DT), [(a,) for a in acc])
    for threshold in (1.0, 2.0, 3.0, 5.0, 8.0):
        online = run(ImuBumpDetector("hand_imu", ImuDetectorConfig(threshold=threshold), DT), [(a,) for a in acc])
        assert first_alarm(stats, threshold, 2) == online.fired_at
    assert np.all(held(stats, 2) <= stats)


# ----- configuration -----


def test_bump_yaml_defaults_to_the_hand_imu_and_rejects_unknown_keys(tmp_path):
    cfg = BumpConfig.from_yaml()
    assert cfg.sensor == "hand_imu"
    assert cfg.imu == ImuDetectorConfig() and cfg.torque == TorqueDetectorConfig() and cfg.tactile == TactileDetectorConfig()
    bad = tmp_path / "bad.yaml"
    bad.write_text("bump_sensor: hand_imu\nimu:\n  noise: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="noise"):
        BumpConfig.from_yaml(bad)
    bad.write_text("bump_sensor: sonar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sonar"):
        BumpConfig.from_yaml(bad)
    good = tmp_path / "torque.yaml"
    good.write_text("bump_sensor: joint_torque\njoint_torque:\n  threshold: 0.8\n", encoding="utf-8")
    cfg = BumpConfig.from_yaml(good)
    assert cfg.sensor == "joint_torque" and cfg.torque.threshold == 0.8 and cfg.torque.gain == TorqueDetectorConfig().gain


# ----- which box: estimated from the drawing and the bins camera -----


def test_the_box_is_the_nearest_in_the_drawing_with_each_tray_where_the_camera_saw_it():
    layout = Layout()
    scene = make_scene(layout)
    names = ("table", "good_wall_ym", "good_wall_yp", "rework_wall_ym")
    region = layout.bin_region("good")
    near_ym = np.array([region.cx, region.cy - 0.085, region.wall_top_z + 0.01])  # 1 cm above the drawn near wall
    assert estimate_box(near_ym, believed_boxes(scene, names))[0] == "good_wall_ym"
    # The camera saw the good tray 15 cm further along +y: the same point is now nearest the table, not that wall.
    moved = believed_boxes(scene, names, {"good": (region.cx, region.cy + 0.15)})
    assert moved["good_wall_ym"].center[1] == pytest.approx(scene.box("good_wall_ym").pos[1] + 0.15)
    assert estimate_box(near_ym, moved)[0] == "table"


# ----- the fake's guarded moves decide by the configured detector -----


def fake_at_home(cfg: BumpConfig | None = None, sensors: SensorConfig | None = None) -> FakeSimulator:
    sim = FakeSimulator(sensors=sensors)
    wrist = HOME_WRIST["left"]
    wall = BoxSpec("wall", (0.30, 0.02, 0.20), (float(wrist[0]) + 0.05, float(wrist[1]) - 0.10, float(wrist[2])), fixed=True)
    table = BoxSpec("table", (1.0, 1.0, 0.1), (0.5, 0.0, 0.4), fixed=True)
    sim.build(SceneSpec(boxes=(wall, table)))
    sim.bump_config = cfg
    return sim


def cross(sim: FakeSimulator) -> float:
    target = HOME_WRIST["left"] + np.array([0.0, -0.2, 0.0])
    return sim.move_wrist("left", target, HOME_QUAT["left"], stop_on=("wall", "table"))


def spike_at(sim: FakeSimulator, at: int, placement: str = "left_hand", size: float = 30.0) -> None:
    """Inject a hand IMU spike ``at`` physics steps from now (after the detector's baseline has filled)."""
    real = sim.step
    count = {"n": 0}

    def step(n: int = 1) -> None:
        for _ in range(n):
            count["n"] += 1
            if count["n"] == at:
                sim.inject_imu(placement, acc=(size, 0.0, 0.0), steps=3)
            real(1)

    sim.step = step


def test_a_guarded_move_halts_when_the_hand_imu_feels_the_injected_bump_and_names_the_box_as_an_estimate():
    sim = fake_at_home()
    sim.inject_bump("wall")
    err = cross(sim)
    assert sim.last_bump == "wall" and err > 0.05  # stopped about halfway
    info = sim.last_bump_info
    assert info["sensor"] == "hand_imu" and info["estimated"] and "drawing" in info["how"]
    assert info["others"]["tactile"] is not None  # the skin felt it too (the torque is still building at the stop)
    assert sim.bump_log == [info]
    assert "estimated" in bump.bump_line(info) and "hand_imu" in bump.bump_line(info)


def test_the_configured_detector_alone_decides_the_stop():
    sim = fake_at_home(BumpConfig(sensor="joint_torque"))
    spike_at(sim, 20)  # an IMU spike with no contact: the torque detector feels nothing
    assert cross(sim) < 1e-9 and sim.last_bump is None
    assert sim.bump_guard.detectors["hand_imu"].fired  # the hand IMU did fire, as a monitor only

    sim = fake_at_home(BumpConfig(sensor="hand_imu"))
    spike_at(sim, 20)
    cross(sim)
    assert sim.last_bump is not None and sim.last_bump_info["sensor"] == "hand_imu"
    assert sim.last_bump_info["others"]["joint_torque"] is None


def test_tactile_and_torque_each_catch_the_injected_bump_when_configured():
    for sensor in ("tactile", "joint_torque"):
        sim = fake_at_home(BumpConfig(sensor=sensor))
        sim.inject_bump("wall")
        cross(sim)
        assert sim.last_bump == "wall" and sim.last_bump_info["sensor"] == sensor


def test_contact_truth_reports_the_true_box_and_says_it_is_truth():
    sim = fake_at_home(BumpConfig(sensor="contact_truth", truth_every=1))
    sim.inject_bump("wall")
    cross(sim)
    assert sim.last_bump == "wall"
    assert not sim.last_bump_info["estimated"] and "TRUTH" in sim.last_bump_info["how"]


def test_a_bump_sensor_the_robot_does_not_carry_is_refused():
    sim = fake_at_home(BumpConfig(sensor="joint_torque"), sensors=SensorConfig(torque_arms=()))
    with pytest.raises(ValueError, match="joint_torque"):
        cross(sim)
    sim = fake_at_home(BumpConfig(sensor="hand_imu"), sensors=SensorConfig.off())
    with pytest.raises(ValueError, match="hand_imu"):
        cross(sim)


def test_a_missed_bump_leaves_the_hand_pressed_short_of_the_target():
    sim = fake_at_home(BumpConfig(sensor="hand_imu", imu=ImuDetectorConfig(threshold=1e6)))
    sim.inject_bump("wall")
    err = cross(sim)
    assert sim.last_bump is None and err > 0.05  # nothing stopped it, and the box stopped the hand


def test_the_fake_estimate_comes_from_the_hand_position_never_from_the_touched_box():
    # The injected bump is on a wall 0.35 m off to the side; the table top is 8.5 cm under the hand. A robot only knows
    # where its hand is, so the estimate must be the table: naming the wall would mean the fake handed it the truth.
    sim = FakeSimulator()
    wrist = HOME_WRIST["left"]
    wall = BoxSpec("wall", (0.30, 0.02, 0.20), (float(wrist[0]) + 0.50, float(wrist[1]) - 0.10, float(wrist[2])), fixed=True)
    table = BoxSpec("table", (1.0, 1.0, 0.1), (0.5, 0.0, float(wrist[2]) - 0.135), fixed=True)
    sim.build(SceneSpec(boxes=(wall, table)))
    sim.inject_bump("wall")
    cross(sim)
    assert sim.last_bump == "table" and sim.last_bump_info["estimated"]
    assert sim.last_bump_info["distance_m"] == pytest.approx(0.085, abs=0.002)


def test_the_estimate_uses_the_robots_bins_seen_when_the_task_gives_it():
    layout = Layout(robot_start=(0.0, 0.0, 0.0))  # already parked at the table: the arms move with the base
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    place_episode(sim, layout, make_episode(layout, 0))
    sim.set_joints(TRAVEL_POSE)
    region = layout.bin_region("good")
    sim.bins_seen = {"good": (region.cx, region.cy)}
    sim.inject_bump("good_wall_ym")
    target = np.array([region.cx - 0.12, region.cy + 0.04, region.wall_top_z + 0.1])
    sim.move_wrist("left", target, HOME_QUAT["left"], stop_on=("table", "good_wall_ym", "good_wall_xm"))
    assert sim.last_bump == "good_wall_ym"
    assert "bins camera" in sim.last_bump_info["how"]


# ----- the panel's windows -----


def test_the_accelerometer_and_joint_torque_windows_show_each_residual_against_its_threshold():
    assert bump.accelerometer_rows(None)[0]["state"] == "idle"
    sim = fake_at_home()
    sim.inject_bump("wall")
    cross(sim)
    acc = bump.accelerometer_rows(sim.bump_guard)
    assert [r["label"] for r in acc] == ["base_imu", "hand_imu", "head_imu", "torso_imu"]
    assert all(set(r) == {"label", "value", "detail", "state"} and r["state"] in {"ok", "warn", "idle"} for r in acc)
    assert all(r["state"] == "idle" for r in acc)  # the move is over: not guarding
    assert "decides the stop" in next(r for r in acc if r["label"] == "hand_imu")["detail"]
    tq = bump.torque_rows(sim.bump_guard)
    assert [r["label"] for r in tq] == [f"left_{i}" for i in range(1, 8)]
    assert all(r["value"].endswith("N*m") for r in tq)


# ----- the Genesis side, without building: the stop path never asks the contact solver -----


def test_genesis_stop_reads_the_detector_and_never_the_contact_solver():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")

    class Kit:
        link_names = ("left_gripper_base", "left_index_1", "left_arm_3")
        held_at = None

        def get_link_pose(self, name, frame=None):
            return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

        def set_joint_positions(self, targets, immediate=False):
            Kit.held_at = dict(targets)

        def get_joint_state(self, names):
            class State:
                pass

            s = State()
            s.names, s.positions = list(names), [0.1] * len(names)
            return s

        class robot:
            @staticmethod
            def get_contacts(*a, **k):
                raise AssertionError("the contact solver is truth only")

    sim = genesis_sim.GenesisSimulator()
    sim._sim = Kit()
    sim._scene = SceneSpec(boxes=(BoxSpec("wall", (0.2, 0.02, 0.2), (0.0, 0.05, 1.0), fixed=True),))
    calm = {p: (np.array([0.0, 0.0, G0]), np.zeros(3)) for p, _ in sim.sensors.placements}
    readings = [calm] * 6 + [{**calm, "left_hand": (np.array([30.0, 0.0, G0]), np.zeros(3))}] * 2
    it = iter(readings)
    sim.read_imus = lambda: next(it)
    sim.read_joint_torques = lambda arm: np.zeros(7)
    sim.torque_model = lambda arm: (np.zeros(7), np.zeros(7))
    sim.read_tactile = lambda arm: {}
    sim.bump_config = BumpConfig()
    guard = BumpGuard(sim, "left", ("wall",), sim.bump_config)
    stops = [sim._stopped_on("left", guard) for _ in range(len(readings))]
    assert stops == [False] * 7 + [True]
    assert sim.last_bump == "wall" and sim.last_bump_info["estimated"]
    assert Kit.held_at is not None and set(Kit.held_at) == {f"left_{i}" for i in range(1, 8)}  # the arm held where it is


# ----- the scorer, on a tiny synthetic probe -----


def write_probe(tmp: Path) -> Path:
    """Two seeds x (2 clean, 2 bumped) crossings in the probe's format: the hand IMU sees a spike at the true contact,
    the torque a build-up, the skin a force."""
    placements = ["left_hand", "head", "base", "torso"]
    lines = []
    for seed in (1, 2):
        for i, clip in enumerate((False, False, True, True)):
            n = 150
            rng = np.random.default_rng([seed, i])
            acc = np.tile([0.0, 0.0, G0], (n, 4, 1)) + rng.normal(0, 0.2, (n, 4, 3))
            acc[:, 0, 0] += lagged(np.concatenate([eased(100, 0.25), np.zeros(n - 100)]))
            external = np.zeros((n, 7))
            tactile = np.zeros((n, 11, 3))
            truth = np.zeros(n, dtype=int)
            if clip:
                k = 70 + 5 * i
                truth[k:] = 1
                acc[k : k + 2, 0, 0] -= 15.0
                external[k:, 1] = np.minimum(np.arange(n - k) * 0.4, 6.0)
                tactile[k:, 2, 2] = 1.5
            tq = torque_trace(n, external, seed=seed * 10 + i)
            arrays = {
                "t": np.arange(n) * DT, "imu_acc": acc, "imu_gyro": np.zeros((n, 4, 3)),
                "torque": np.array([s[0] for s in tq]), "dp": np.array([s[1] for s in tq]), "bias": np.array([s[2] for s in tq]),
                "tactile": tactile, "fingertips": np.zeros((n, 5)), "truth_hand": truth, "truth_part": np.zeros(n, dtype=int),
                "truth_box": np.where(truth > 0, 0, -1), "wrist": np.zeros((n, 3)), "phase": np.zeros(n, dtype=int),
            }
            arrays["dp"][0] = np.nan  # the first step has no previous velocity
            name = f"trial_s{seed}_{i}.npz"
            np.savez(tmp / name, **arrays)
            lines.append(json.dumps({"seed": seed, "file": name, "dt": DT, "arm": "left", "placements": placements, "clip": clip, "speed_mps": 0.25}))
    (tmp / "trials.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp


def test_the_scorer_picks_on_one_seed_and_scores_on_the_other(tmp_path, capsys):
    probe = write_probe(tmp_path)
    assert score_mod.main([str(probe), "--pick-seeds", "1", "--score-seeds", "2", "--margin", "2"]) == 0  # 2 clean picks: margin for the noise
    result = json.loads((probe / "score.json").read_text(encoding="utf-8"))
    hand = result["sensors"]["hand_imu"]
    assert hand["bumps"] == 2 and hand["caught"] == 2 and hand["missed"] == 0 and hand["clean_moves"] == 2
    assert hand["latency_ms_median"] == pytest.approx(10.0)  # hold 2: the second spike sample
    assert result["sensors"]["joint_torque"]["caught"] == 2 and result["sensors"]["tactile"]["latency_ms_max"] == 0.0
    assert result["sensors"]["head_imu"]["missed"] == 2  # no spike reached the head
    assert result["sensors"]["contact_truth"]["truth_only"] and result["winner"] != "contact_truth"
    assert result["winner"] in {"hand_imu", "tactile", "joint_torque"}
    out = capsys.readouterr().out
    assert "hand_imu" in out and "winner:" in out


def test_the_scorer_refuses_overlapping_seeds_and_picks_within_the_budget(tmp_path):
    probe = write_probe(tmp_path)
    with pytest.raises(SystemExit, match="overlap"):
        score_mod.main([str(probe), "--pick-seeds", "1-2", "--score-seeds", "2"])
    assert score_mod.pick_threshold([0.5, 2.0, 1.0], budget=0.0, margin=1.0) == 2.0  # none of 3 may be over
    assert score_mod.pick_threshold([0.5, 2.0, 1.0] * 34, budget=0.01, margin=1.0) == 2.0  # 1 of 102 may: still a 2.0
    peaks = [float(v) for v in range(1, 201)]
    assert score_mod.pick_threshold(peaks, budget=0.01, margin=1.0) == 198.0  # 2 of 200 over 198
    assert score_mod.parse_seeds("1-3,7") == [1, 2, 3, 7]
