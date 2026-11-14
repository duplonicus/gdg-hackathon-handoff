"""The robot's sensor layer: the part specs, the config wiring, and the fake's scripted readings (no GPU)."""

from pathlib import Path

import numpy as np
import pytest

from handoff.sim.base import (
    DEFAULT_SENSORS,
    G0,
    BoxSpec,
    FingertipSpec,
    ImuSpec,
    SceneSpec,
    SensorConfig,
    SkinSpec,
    fingertip_links,
    skin_links,
)
from handoff.sim.fake import HOLD_OFFSET, HOME_WRIST, FakeSimulator

ROOT = Path(__file__).resolve().parents[2]
PART = 0.04  # m, the cube the fake hand closes on
STATES = {"ok", "warn", "idle"}


def fake_with_part(sensors: SensorConfig | None = None) -> FakeSimulator:
    """A fake with one 4 cm part sitting at the left hand's hold point."""
    pos = tuple(float(v) for v in HOME_WRIST["left"] + HOLD_OFFSET)
    sim = FakeSimulator(sensors=sensors)
    sim.build(SceneSpec(boxes=(BoxSpec("part", (PART, PART, PART), pos), BoxSpec("table", (1.0, 1.0, 0.1), (0.5, 0.0, 0.4), fixed=True))))
    return sim


# ----- the part specs -----


def test_imu_noise_is_the_zed_x_density_per_root_hz_times_root_sample_rate():
    acc, gyro = ImuSpec().sigma(0.01)
    assert acc == pytest.approx(2.3e-3 * G0 * 10.0)  # 2.3 mg/sqrt(Hz) at 100 samples/s
    assert gyro == pytest.approx(np.deg2rad(0.20) * 10.0)
    assert ImuSpec().acc_range == pytest.approx(12 * G0) and ImuSpec().gyro_range == pytest.approx(np.deg2rad(1000))


def test_fsr_reads_nothing_below_its_actuation_force_and_clips_at_its_range():
    spec = FingertipSpec()
    assert spec.reading(0.19) == 0.0
    assert spec.reading(0.5) == pytest.approx(0.5)
    assert spec.reading(25.0) == 20.0
    assert spec.reading(10.0, gain=1.06) == pytest.approx(10.6)  # a part-to-part gain of +1 sigma
    assert spec.reading(10.0, z=1.0) == pytest.approx(10.2)  # +1 sigma of single-part repeatability
    assert spec.reading(0.21, gain=0.9) == 0.0  # a low-gain part does not trip at 0.21 N true


def test_skin_noise_is_proportional_so_an_untouched_link_reads_zero():
    spec = SkinSpec()
    np.testing.assert_array_equal(spec.reading(np.zeros(3), np.array([3.0, -3.0, 3.0])), np.zeros(3))
    np.testing.assert_allclose(spec.reading(np.array([1.0, -2.0, 0.0]), np.array([1.0, 1.0, 0.0])), [1.025, -2.05, 0.0])
    np.testing.assert_allclose(spec.reading(np.array([0.0, 0.0, -9.0])), [0.0, 0.0, -4.4])  # the A201 1 lb range


def test_hand_link_names_are_the_kits():
    assert skin_links("left") == (
        "left_gripper_base", "left_index_1", "left_index_2", "left_middle_1", "left_middle_2", "left_ring_1",
        "left_ring_2", "left_little_1", "left_little_2", "left_thumb_1", "left_thumb_2",
    )
    assert fingertip_links("left")["left_thumb_tip"] == ("left_thumb_tip", "left_thumb_force_sensor")
    assert sorted(fingertip_links("left")) == sorted(f"left_{f}_tip" for f in ("index", "middle", "ring", "little", "thumb"))


def test_every_sensor_link_exists_in_the_robot_usd():
    """Each link a sensor goes on is a rigid body of the MH3 USD, by the kit's public name (catches a typo before a
    two-minute Genesis build would)."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    from mh3_genesis.simulator import _public_link_name

    stage = Usd.Stage.Open(str(ROOT / "assets" / "MH3.usd"))
    links = {_public_link_name(p.GetName()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    cfg = SensorConfig.from_yaml(DEFAULT_SENSORS)
    wanted = {link for _, link in cfg.placements} | set(skin_links("left"))
    wanted |= {link for pair in fingertip_links("left").values() for link in pair}
    assert wanted - links == set()


# ----- config wiring -----


def test_sensors_yaml_is_on_with_the_four_imu_placements_and_the_left_hand():
    cfg = SensorConfig.from_yaml(DEFAULT_SENSORS)
    assert cfg.enabled
    assert dict(cfg.placements) == {
        "left_hand": "left_gripper_base", "head": "head_camera_mount", "base": "mobile_base", "torso": "torso_base",
    }
    assert cfg.arms("torque") == cfg.arms("tactile") == cfg.arms("fingertip") == ("left",)
    assert cfg.imu == ImuSpec()  # the YAML never overrides a datasheet number


def test_sensors_yaml_rejects_unknown_keys_and_can_turn_everything_off(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("enabled: true\nacc_noise: 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="acc_noise"):
        SensorConfig.from_yaml(bad)
    off = tmp_path / "off.yaml"
    off.write_text("enabled: false\n", encoding="utf-8")
    cfg = SensorConfig.from_yaml(off)
    assert cfg.placements == () and cfg.arms("torque") == ()


def test_genesis_simulator_loads_the_sensor_config_without_building():
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")
    assert genesis_sim.GenesisSimulator().sensors == SensorConfig.from_yaml(DEFAULT_SENSORS)
    assert not genesis_sim.GenesisSimulator(sensors=None).sensors.enabled
    custom = SensorConfig(imus=(("head", "head_camera_mount"),))
    assert genesis_sim.GenesisSimulator(sensors=custom).sensors is custom


def test_sensors_off_reads_empty_on_the_fake():
    sim = fake_with_part(SensorConfig.off())
    assert sim.read_imus() == {} and sim.read_tactile("left") == {} and sim.read_fingertips("left") == {}
    assert sim.read_joint_torques("left").shape == (0,)
    assert sim.sensor_windows() == {}


# ----- the fake's readings -----


def test_fake_imus_read_gravity_up_at_rest_and_the_base_turn_rate():
    sim = fake_with_part()
    imus = sim.read_imus()
    assert set(imus) == {"left_hand", "head", "base", "torso"}
    for acc, gyro in imus.values():
        np.testing.assert_allclose(acc, [0.0, 0.0, G0])
        np.testing.assert_allclose(gyro, 0.0)
    sim.drive(0.0, 0.5)
    assert sim.read_imus()["base"][1][2] == pytest.approx(0.5)


def test_an_injected_imu_spike_lasts_its_steps_and_reading_does_not_step():
    sim = fake_with_part()
    sim.inject_imu("left_hand", acc=(30.0, 0.0, 0.0), steps=3)
    t0 = sim.time
    for _ in range(5):
        assert sim.read_imus()["left_hand"][0][0] == pytest.approx(30.0)
    assert sim.time == t0
    assert sim.read_imus()["head"][0][0] == 0.0  # only that placement
    sim.step(2)
    assert sim.read_imus()["left_hand"][0][0] == pytest.approx(30.0)
    sim.step(1)
    assert sim.read_imus()["left_hand"][0][0] == 0.0
    with pytest.raises(KeyError):
        sim.inject_imu("right_hand", acc=(1.0, 0.0, 0.0))


def test_an_imu_clips_at_its_range():
    sim = fake_with_part()
    sim.inject_imu("base", acc=(500.0, 0.0, 0.0), gyro=(0.0, 0.0, 100.0))
    acc, gyro = sim.read_imus()["base"]
    assert acc[0] == pytest.approx(12 * G0) and gyro[2] == pytest.approx(np.deg2rad(1000))


def test_fingertips_press_once_the_closure_squeezes_past_the_part_width():
    sim = fake_with_part()
    sim.set_hand("left", 0.5)  # gap 5 cm > 4 cm part
    assert set(sim.read_fingertips("left").values()) == {0.0}
    sim.set_hand("left", 0.603)  # 0.3 mm inside the part: 0.06 N true, under the FSR's 0.2 N
    assert set(sim.read_fingertips("left").values()) == {0.0}
    sim.set_hand("left", 0.7)  # 1 cm inside: 2 N
    tips = sim.read_fingertips("left")
    assert set(tips) == set(fingertip_links("left")) and all(v == pytest.approx(2.0) for v in tips.values())
    sim.set_hand("left", 0.9)
    assert sim.read_fingertips("left")["left_index_tip"] == pytest.approx(6.0)
    sim.set_hand("left", 1.0)
    sim.set_object_pose("part", (2.0, 2.0, 0.5))  # nothing in the hand: nothing to press on
    assert set(sim.read_fingertips("left").values()) == {0.0}
    assert sim.read_fingertips("right") == {}  # no fingertip sensors on the right hand


def test_scripted_fingertips_still_pass_through_the_fsr():
    sim = fake_with_part()
    sim.set_fingertips("left", {"left_thumb_tip": 30.0, "left_index_tip": 0.1})
    tips = sim.read_fingertips("left")
    assert tips["left_thumb_tip"] == 20.0 and tips["left_index_tip"] == 0.0 and tips["left_ring_tip"] == 0.0
    sim.set_fingertips("left", None)
    sim.set_hand("left", 0.7)
    assert sim.read_fingertips("left")["left_thumb_tip"] == pytest.approx(2.0)
    with pytest.raises(KeyError):
        sim.set_fingertips("left", {"left_pinky_tip": 1.0})


def test_tactile_skin_and_joint_torques_are_scripted_and_ranged():
    sim = fake_with_part()
    skin = sim.read_tactile("left")
    assert set(skin) == set(skin_links("left")) and all(np.all(f == 0.0) for f in skin.values())
    sim.set_tactile("left", "left_gripper_base", (0.0, 0.0, 9.0))
    np.testing.assert_allclose(sim.read_tactile("left")["left_gripper_base"], [0.0, 0.0, 4.4])  # A201 1 lb range
    with pytest.raises(KeyError):
        sim.set_tactile("left", "left_index_tip", (1.0, 0.0, 0.0))  # a tip is the FSR's, not the skin's
    np.testing.assert_allclose(sim.read_joint_torques("left"), np.zeros(7))
    sim.set_joint_torques("left", np.arange(7.0))
    np.testing.assert_allclose(sim.read_joint_torques("left"), np.arange(7.0))
    assert sim.read_joint_torques("right").shape == (0,)
    with pytest.raises(ValueError):
        sim.set_joint_torques("left", [1.0, 2.0])
    sim.reset()
    np.testing.assert_allclose(sim.read_joint_torques("left"), np.zeros(7))


def test_truth_queries_answer_per_box_and_per_link():
    sim = fake_with_part()
    assert sim.truth_hand_contacts("left", ("part", "table")) == {"part": 1, "table": 0}
    sim.set_hand("left", 1.0)
    assert sim.truth_hand_contacts("left", ("part",)) == {"part": 4}
    assert sim.truth_finger_penetration("left", "part") == {}
    sim.set_finger_penetration("left", "part", {"left_index_tip": 0.002})
    assert sim.truth_finger_penetration("left", "part") == {"left_index_tip": 0.002}


# ----- what the panel shows -----


def test_sensor_windows_show_every_sensor_in_the_set_sensor_contract():
    sim = fake_with_part()
    sim.set_hand("left", 0.7)
    sim.inject_imu("left_hand", acc=(500.0, 0.0, 0.0))
    windows = sim.sensor_windows()
    assert list(windows) == ["IMUs (ZED X IMU)", "Left arm joint torque", "Left hand tactile skin", "Left fingertips"]
    for rows in windows.values():
        for row in rows:
            assert set(row) == {"label", "value", "detail", "state"} and row["state"] in STATES
    imus = {r["label"]: r for r in windows["IMUs (ZED X IMU)"]}
    assert imus["left_hand"]["state"] == "warn" and imus["head"]["state"] == "ok"
    assert [r["label"] for r in windows["Left arm joint torque"]] == [f"left_{i}" for i in range(1, 8)]
    assert {r["state"] for r in windows["Left hand tactile skin"]} == {"idle"}
    tips = {r["label"]: r for r in windows["Left fingertips"]}
    assert tips["thumb_tip"]["value"] == "2.00 N" and tips["thumb_tip"]["state"] == "ok"


def test_skin_clips_on_the_force_magnitude_not_per_axis():
    """One A201 element senses one force: a slanted 6.9 N push reads 4.4 N, never 4.4 * sqrt(3)."""
    reading = SkinSpec().reading(np.array([4.0, 4.0, 4.0]))
    assert float(np.linalg.norm(reading)) == pytest.approx(4.4)
    np.testing.assert_allclose(reading / np.linalg.norm(reading), np.ones(3) / np.sqrt(3))  # direction kept
    sim = fake_with_part()
    sim.set_tactile("left", "left_index_1", (3.0, 3.0, 3.0))
    rows = {r["label"]: r for r in sim.sensor_windows()["Left hand tactile skin"]}
    assert rows["index_1"]["value"] == "4.40 N" and rows["index_1"]["state"] == "warn"


def test_a_kit_step_hook_reads_each_steps_fingertips_inside_a_multi_step_call():
    """The demo's panel hooks run inside the kit's step; a read there must see that step, not a cache from before the
    call (a stale step count froze the Grip force window for a whole settle)."""
    genesis_sim = pytest.importorskip("handoff.sim.genesis_sim")

    class Kit:
        def __init__(self):
            self.n, self.hooks = 0, []

        def step(self):
            self.n += 1
            for hook in self.hooks:
                hook(self)

    class Pad:
        def __init__(self, kit, scale):
            self.kit, self.scale = kit, scale

        def read(self):
            return np.array([0.0, 0.0, self.scale * (1.0 + self.kit.n)])

    sim = genesis_sim.GenesisSimulator(sensors=SensorConfig())
    kit = Kit()
    sim._sim = kit
    tip = "left_index_tip"
    sim._tips = {"left": {tip: (Pad(kit, 1.0), Pad(kit, 0.0))}}
    sim._pad_in_tip = {tip: np.array([1.0, 0.0, 0.0, 0.0])}
    sim._tip_gain = {tip: 1.0}
    sim._noise_rng = type("NoNoise", (), {"standard_normal": staticmethod(lambda *a: np.zeros(*a) if a else 0.0)})()
    seen = []
    kit.hooks.append(lambda _k: seen.append(sim.read_fingertips("left")[tip]))
    assert sim.read_fingertips("left")[tip] == pytest.approx(1.0)  # read before the call: step 0
    sim.step(3)
    assert seen == pytest.approx([2.0, 3.0, 4.0])
    assert sim.read_fingertips("left")[tip] == pytest.approx(4.0)
