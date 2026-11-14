"""The second pass of the force grip, after the review's Genesis CPU run (Oct 21): the command never runs far ahead of
the fingers, a grip keeper holds the force through the carry, the fixed closure stays the default until the GPU probe
says otherwise. Also the sensor fixes from the same review: every sensor on the panel, the skin as one force per
element, the bump guard on an arm without the configured sensor, and holding() from the fingertips."""

import numpy as np
import pytest

from handoff.bump import BumpConfig, BumpGuard, TactileBumpDetector, TactileDetectorConfig, bump_line
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sensors import sensor_window_hook
from handoff.sim.base import SensorConfig
from handoff.sim.fake import FakeSimulator
from handoff.skills import (
    EMPTY_CLOSURE,
    GRIP_MAX_LEAD,
    SKILLS,
    TRAVEL_POSE,
    SkillResult,
    TaskContext,
    finger_side_n,
    grip_keeper,
    grip_rows,
    grip_target_n,
    holding,
    keeper_line,
    let_go,
    open_hand,
)
from handoff.status import RunStatus

FINGER_TIPS = ("left_index_tip", "left_middle_tip", "left_ring_tip", "left_little_tip")


class Blocked(FakeSimulator):
    """Fingers that stop where they meet the part (``block``) and press harder the further the command is past it,
    30 N per unit of closure on every tip: the soft contact the fake otherwise leaves out. ``block = 1`` is no part."""

    block = 0.65
    zero_after: int | None = None  # every read after this many reads is 0 N (the part gave way under the sensors)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.closures: list[float] = []
        self.reads = 0

    def set_hand(self, arm, closed):
        self.closures.append(float(closed))
        super().set_hand(arm, closed)

    def hand_closure(self, arm):
        return min(super().hand_closure(arm), self.block)

    def read_fingertips(self, arm):
        self.reads += 1
        force = max(0.0, (self._closure[arm] - self.block) * 30.0)
        if self.zero_after is not None and self.reads > self.zero_after:
            force = 0.0
        return {**dict.fromkeys(FINGER_TIPS, force), "left_thumb_tip": force}


def at_the_part(sim: FakeSimulator, control: str = "force") -> TaskContext:
    layout = Layout()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 0)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, grasp_control=control)
    for name in ("drive_to_table", "look_at_table", "reach", "approach"):
        assert SKILLS[name](sim, ctx).ok, name
    return ctx


def test_the_fixed_closure_stays_the_default_until_the_probe_picks():
    ctx = TaskContext(layout=Layout(), target_part="part_00")
    assert ctx.grasp_control == "fixed"
    sim = FakeSimulator()
    ctx = at_the_part(sim, control=ctx.grasp_control)
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok and result.data["control"] == "fixed"
    assert grip_keeper(sim, "left") is None  # no keeper on a fixed grip


def test_the_command_never_runs_more_than_the_lead_ahead_of_fingers_the_part_stops():
    sim = Blocked()
    ctx = at_the_part(sim)
    sim.closures.clear()
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok, result.reason
    assert max(sim.closures) <= sim.block + GRIP_MAX_LEAD + 1e-9  # the old ramp ran 0.08 ahead and overshot to 14 N
    assert result.data["finger_side_n"] >= grip_target_n(ctx.layout)
    assert result.data["finger_side_n"] <= 4 * GRIP_MAX_LEAD * 30.0 + 1e-9  # bounded by the lead, not 15x the target
    assert grip_keeper(sim, "left") is not None and "keeper" in result.data


def test_the_keeper_closes_further_when_the_part_gives_and_holds_the_force_again():
    sim = Blocked()
    ctx = at_the_part(sim)
    assert SKILLS["grasp"](sim, ctx).ok
    keeper = grip_keeper(sim, "left")
    target = grip_target_n(ctx.layout)
    sim.block = 0.70  # the contact settles 0.05 deeper: the held command no longer presses (the Genesis lift drop)
    assert finger_side_n(sim.read_fingertips("left"), "left") == 0.0
    sim.step(200)
    assert keeper.tightened > 0 and not keeper.lost
    assert finger_side_n(sim.read_fingertips("left"), "left") >= target
    assert keeper.cmd <= sim.block + GRIP_MAX_LEAD + 1e-9


def test_the_keeper_marks_the_part_lost_when_the_fingers_close_past_where_a_part_could_be():
    sim = Blocked()
    ctx = at_the_part(sim)
    assert SKILLS["grasp"](sim, ctx).ok
    keeper = grip_keeper(sim, "left")
    sim.block = 1.0  # the part is gone: nothing stops the fingers, nothing presses
    sim.step(400)
    assert keeper.lost
    assert EMPTY_CLOSURE <= sim.hand_closure("left") < EMPTY_CLOSURE + GRIP_MAX_LEAD + 0.01  # stopped tightening there
    assert sim.closures[-1] > 0.5  # it never opens the hand on its own
    assert "LOST" in keeper_line("part_00", keeper.summary())
    assert grip_rows(sim.read_fingertips("left"), 0.93, sim.hand_closure("left"), keeper=keeper)[2]["value"] == "LOST"


def test_a_grip_that_does_not_survive_the_hold_fails_and_takes_the_keeper_off():
    sim = Blocked()
    ctx = at_the_part(sim)
    sim.reads = 0
    sim.zero_after = 40  # past where the ramp reaches the force, every read is 0 N and the fingers cannot close further
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.reason.startswith("grip did not hold: 0.00 of 0.93 N on the finger tips after the 0.2 s hold")
    assert grip_keeper(sim, "left") is None


def test_opening_the_hand_takes_the_keeper_off_first_so_it_cannot_close_again():
    sim = Blocked()
    ctx = at_the_part(sim)
    assert SKILLS["grasp"](sim, ctx).ok
    keeper = open_hand(sim, ctx)
    assert keeper is not None and grip_keeper(sim, "left") is None
    sim.step(100)
    assert sim.closures[-1] == 0.0  # nothing closed it again on the falling force
    assert keeper_line("part_00", keeper.summary()).startswith("grip keeper part_00: held to the release")


def test_let_go_and_reset_and_a_fixed_grasp_each_take_the_keeper_off():
    sim = Blocked()
    ctx = at_the_part(sim)
    assert SKILLS["grasp"](sim, ctx).ok
    assert let_go(sim, "left") is not None and grip_keeper(sim, "left") is None and sim.closures[-1] == 0.0
    assert SKILLS["grasp"](sim, ctx).ok and grip_keeper(sim, "left") is not None
    ctx.grasp_control = "fixed"
    SKILLS["grasp"](sim, ctx)
    assert grip_keeper(sim, "left") is None
    ctx.grasp_control = "force"
    let_go(sim, "left")
    assert SKILLS["grasp"](sim, ctx).ok
    sim.reset()
    assert grip_keeper(sim, "left") is None


def test_the_force_grip_carries_the_part_through_the_whole_pick_and_place_on_the_fake():
    sim = FakeSimulator()
    ctx = at_the_part(sim)
    for name in ("grasp", "lift", "move_to_bin", "lower", "release"):
        result = SKILLS[name](sim, ctx)
        assert result.ok, (name, result.reason)
    assert result.data["grip_keeper"]["lost"] is False  # release reports how the keeper held
    assert grip_keeper(sim, "left") is None


def test_holding_reads_the_fingertips_not_the_contact_solver():
    sim = FakeSimulator()
    at = at_the_part(sim)

    def no_truth(*_a, **_k):
        raise AssertionError("holding() read the contact solver on a hand with fingertip sensors")

    sim.hand_contacts = no_truth
    assert holding(sim, at) is False  # open hand, nothing on the tips
    sim.set_fingertips("left", {"left_index_tip": 0.5})  # a part stuck to the open hand
    assert holding(sim, at) is True
    off = FakeSimulator(sensors=SensorConfig.off())
    ctx = at_the_part(off, control="fixed")
    assert holding(off, ctx) is (off.hand_contacts("left", ctx.target_part) > 0)  # no fingertips: falls back


def test_every_carried_sensor_gets_a_panel_window_every_ten_steps():
    sim = FakeSimulator()
    sim.build(make_scene(Layout()))
    status = RunStatus("oracle")
    hook = sensor_window_hook(sim, status, every=10, skip=("fingertips",))
    for _ in range(9):
        hook(None)
    assert status.snapshot()["sensors"] == []
    hook(None)
    names = [w["name"] for w in status.snapshot()["sensors"]]
    assert names == [n for n in sim.sensor_windows() if "fingertips" not in n]
    assert any("IMU" in n for n in names) and any("joint torque" in n for n in names) and any("skin" in n for n in names)


def test_a_failing_sensor_read_turns_the_windows_off_with_one_log_line(tmp_path):
    sim = FakeSimulator()
    sim.build(make_scene(Layout()))
    status = RunStatus("oracle", tmp_path / "run.log")

    def broken():
        raise RuntimeError("bus timeout")

    sim.sensor_windows = broken
    hook = sensor_window_hook(sim, status, every=1)
    hook(None)
    hook(None)
    lines = status.snapshot()["lines"]
    assert sum("sensor windows: display off" in str(line) for line in lines) == 1


def test_the_skin_detector_sees_force_magnitude_only_like_a_real_element():
    frames = []
    for k in range(80):
        f = np.zeros((11, 3))
        f[2] = [1.0, 0.0, 0.0] if k < 40 else [0.0, 1.0, 0.0]  # same 1 N, turned 90 degrees: no new force on an A201
        frames.append((f,))
    det = TactileBumpDetector("tactile", TactileDetectorConfig(), 0.01)
    for sample in frames:
        det.update(*sample)
    assert det.fired_at is None


def test_a_guarded_move_on_an_arm_without_the_configured_sensor_falls_back_and_says_so():
    sim = FakeSimulator()
    sim.build(make_scene(Layout()))
    guard = BumpGuard(sim, "right", ("table",), BumpConfig())  # hand_imu: only the left hand carries one
    assert guard.fallback_from == "hand_imu"
    assert guard.sensor in ("torso_imu", "head_imu", "base_imu")  # the right arm has no torque or skin either
    info = guard.report(np.zeros((1, 3)), sim._scene, None, 0.0)
    assert info["fallback_from"] == "hand_imu"
    assert "hand_imu is not carried on this arm" in bump_line(info)


def test_the_sort_context_passes_a_set_closure_to_every_pick_and_leaves_the_default_otherwise():
    from handoff.tasks.sort import SortContext

    layout = Layout()
    episode = make_episode(layout, 0)
    ctx = SortContext(layout=layout, episode=episode, ladder=None, close_fraction=0.75)
    assert ctx.task_ctx(episode.parts[0]).close_fraction == 0.75
    plain = SortContext(layout=layout, episode=episode, ladder=None)
    assert plain.task_ctx(episode.parts[0]).close_fraction == TaskContext(layout=layout, target_part="p").close_fraction


def test_the_palm_requirement_waits_for_the_palm_skin_and_is_skipped_without_skin():
    from handoff.skills import palm_ok

    sim = FakeSimulator()
    sim.build(make_scene(Layout()))
    assert palm_ok(sim, "left", None)
    sim.set_tactile("left", "left_gripper_base", np.array([0.0, 0.0, 0.3]))
    assert not palm_ok(sim, "left", 1.0)
    sim.set_tactile("left", "left_gripper_base", np.array([0.0, 0.0, 1.5]))
    assert palm_ok(sim, "left", 1.0)
    bare = FakeSimulator(sensors=SensorConfig.off())
    bare.build(make_scene(Layout()))
    assert palm_ok(bare, "left", 1.0)  # no skin to ask


def test_a_hand_without_a_separate_thumb_skips_the_thumb_step():
    sim = FakeSimulator()
    ctx = at_the_part(sim)
    ctx.grip_thumb_n = 5.0
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok and result.data["thumb"]["separate"] is False
    assert grip_keeper(sim, "left").thumb_cmd is None


# ----- the touch grip (--grip touch): fingertips feel the part, then the weld holds it (Oct 21) -----


def test_the_touch_grip_halts_at_the_part_welds_it_and_takes_the_keeper_off():
    from handoff.skills import TOUCH_THUMB_N, grip_settings

    settings = grip_settings("touch")
    assert settings == {"grip": "touch", "grasp_control": "force", "grasp_mode": "weld", "grip_thumb_n": TOUCH_THUMB_N}
    sim = Blocked()
    ctx = at_the_part(sim)
    ctx.grasp_mode, ctx.grip_thumb_n = settings["grasp_mode"], settings["grip_thumb_n"]
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok and result.data["welded"], result.reason
    assert f"attach:{ctx.target_part}" in sim.calls
    assert grip_keeper(sim, "left") is None  # the weld holds the part; nothing squeezes on
    assert max(sim.closures) <= Blocked.block + GRIP_MAX_LEAD + 1e-9  # never commanded deep into the part
    for name in ("lift", "move_to_bin", "lower", "release"):
        r = SKILLS[name](sim, ctx)
        assert r.ok, (name, r.reason)
    assert f"detach:{ctx.target_part}" in sim.calls


def test_the_grip_log_line_names_the_touch_grip_and_its_weld():
    """A touch grip read "force ... 0.62 of 0.93 N ..., held" in the Oct 23 demo log: a squeeze that fell short and
    passed anyway, to anyone reading it (Oct 26 nightly review). It names the touch grip, the weld and the thumb."""
    from handoff.skills import grip_line, grip_settings

    sim = Blocked()
    ctx = at_the_part(sim)
    for key, value in grip_settings("touch").items():
        setattr(ctx, key, value)
    result = SKILLS["grasp"](sim, ctx)
    line = grip_line(ctx.target_part, result)
    assert line.startswith(f"grip {ctx.target_part}: touch (welded), closure ")
    assert "squeeze not needed under the weld" in line and " of 0.93 N" not in line
    assert ", thumb closed with the fingers (no separate thumb)" in line  # the fake has no separate thumb
    # a separate thumb (the real hand) shows its own force against its target, and says when it fell short
    data = {**result.data, "thumb": {"separate": True, "tip_n": 0.54, "target_n": 0.5, "reached": True}}
    assert ", thumb on its own 0.54 of 0.50 N, " in grip_line("part_15", SkillResult(True, "", data))
    data["thumb"] = {"separate": True, "tip_n": 0.31, "target_n": 0.5, "reached": False}
    assert ", thumb on its own 0.31 of 0.50 N (short), " in grip_line("part_15", SkillResult(True, "", data))
    # a force close welded without the thumb step (--grasp weld --grip force) is not called the touch grip
    plain = {**{k: v for k, v in result.data.items() if k != "thumb"}, "grip": "force"}
    assert grip_line("part_15", SkillResult(True, "", plain)).startswith("grip part_15: force (welded), ")
    # a weld after a fixed close is named too
    assert grip_line("part_15", SkillResult(True, "", {"control": "fixed", "welded": True})).startswith("grip part_15: fixed (welded), ")


def test_a_failed_touch_grip_is_still_named_the_touch_grip():
    """The grip is named by what was asked for, not guessed from the result: a touch grip that fails is never welded,
    and its line read "force" (Oct 27 nightly review, round 2 of the Oct 26 finding)."""
    from handoff.skills import grip_line, grip_settings

    sim = Blocked()
    sim.block = 1.0  # no part: the fingers close on the hand itself
    ctx = at_the_part(sim)
    for key, value in grip_settings("touch").items():
        setattr(ctx, key, value)
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok and not result.data.get("welded")
    line = grip_line(ctx.target_part, result)
    assert line.startswith(f"grip {ctx.target_part}: touch, closure "), line
    assert "FAILED: hand closed without touching the part" in line
    # the force grip, asked for by name, is still "force"
    sim = Blocked()
    sim.block = 1.0
    ctx = at_the_part(sim)
    for key, value in grip_settings("force").items():
        setattr(ctx, key, value)
    assert grip_line(ctx.target_part, SKILLS["grasp"](sim, ctx)).startswith(f"grip {ctx.target_part}: force, closure ")


def test_grip_settings_cover_every_grip_and_the_sort_context_passes_the_thumb():
    from handoff.skills import grip_settings
    from handoff.tasks.sort import SortContext

    assert grip_settings("fixed") == {"grip": "fixed", "grasp_control": "fixed"}
    assert grip_settings("force") == {"grip": "force", "grasp_control": "force"}
    with pytest.raises(ValueError):
        grip_settings("squeeze")
    layout = Layout()
    episode = make_episode(layout, 0)
    ctx = SortContext(layout=layout, episode=episode, ladder=None, **grip_settings("touch"))
    tctx = ctx.task_ctx(episode.parts[0])
    assert (tctx.grip, tctx.grasp_control, tctx.grasp_mode, tctx.grip_thumb_n) == ("touch", "force", "weld", grip_settings("touch")["grip_thumb_n"])


def test_a_welded_part_counts_as_in_hand_without_finger_contact():
    """The touch grip rests its fingers at the surface: the contact solver may read nothing while the weld carries the
    part, and the lift must not call that a drop (3 of 30 touch lifts did, Oct 21)."""
    from handoff.checks import snapshot
    from handoff.skills import part_in_hand

    sim = Blocked()
    ctx = at_the_part(sim)
    ctx.grasp_mode = "weld"
    assert SKILLS["grasp"](sim, ctx).ok and SKILLS["lift"](sim, ctx).ok
    state = snapshot(sim, ctx.layout, (ctx.arm,), ctx.parts_in_play)
    state.contacts.clear()  # the fingers rest at the surface: no contact reported
    assert part_in_hand(state, ctx)
    ctx.grasp_mode = "fingers"
    assert not part_in_hand(state, ctx)  # without the weld, no contact means not held
