"""The force-controlled grasp on the kinematic fake: the hand closes until the fingertips feel the grip force, not to
a position inside the part."""

import pytest

from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.base import SensorConfig
from handoff.sim.fake import OPEN_APERTURE_M, FakeSimulator
from handoff.skills import (
    EMPTY_CLOSURE,
    SKILLS,
    TRAVEL_POSE,
    SkillResult,
    TaskContext,
    finger_side_n,
    grip_budget,
    grip_line,
    grip_rows,
    grip_target_n,
)

FINGER_TIPS = ("left_index_tip", "left_middle_tip", "left_ring_tip", "left_little_tip")


class Recording(FakeSimulator):
    """The fake, keeping every hand command unrounded (its own call log rounds to 0.01)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.closures: list[float] = []

    def set_hand(self, arm, closed):
        self.closures.append(float(closed))
        super().set_hand(arm, closed)


def at_the_part(sim: Recording | None = None, seed: int = 0, control: str = "force") -> tuple[Recording, TaskContext]:
    """The open hand beside the part, ready to close (force-controlled unless ``control`` says otherwise)."""
    layout = Layout()
    sim = sim or Recording()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, grasp_control=control)
    for name in ("drive_to_table", "look_at_table", "reach", "approach"):
        assert SKILLS[name](sim, ctx).ok, name
    return sim, ctx


def commanded(sim: Recording) -> list[float]:
    return list(sim.closures)


def test_the_grip_target_comes_from_the_part_mass_and_friction():
    budget = grip_budget(Layout())
    assert budget["mass_kg"] == pytest.approx(0.0512)  # 4 x 4 x 8 cm at 400 kg/m^3
    assert budget["hold_n"] == pytest.approx(2.5 * 0.0512 * 9.80665 / (2 * 1.5))  # 0.42 N
    assert budget["slide_floor_n"] == pytest.approx(1.5 * 0.0512 * 9.80665 * 1.24)  # 0.93 N: above any sliding push
    assert budget["target_n"] == pytest.approx(budget["slide_floor_n"])
    heavy = grip_budget(Layout(part_rho=4000.0))
    assert heavy["target_n"] == pytest.approx(10 * budget["target_n"])  # scales with the part, not a tuned constant


def test_the_ramp_ends_at_the_grip_force_well_short_of_the_old_closure():
    sim, ctx = at_the_part()
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok, result.reason
    d = result.data
    target = grip_target_n(ctx.layout)
    assert d["control"] == "force" and d["target_n"] == pytest.approx(target)
    assert d["finger_side_n"] >= target
    # The fake's tips first touch the 4 cm part at closure 1 - 0.04 / OPEN_APERTURE_M = 0.6; the grip is just past it.
    touch = 1.0 - 0.04 / OPEN_APERTURE_M
    assert touch < d["closure"] < touch + 0.03
    assert max(commanded(sim)) == pytest.approx(d["closure_cmd"])  # never commanded past where the force was reached
    assert max(commanded(sim)) < ctx.close_fraction
    assert len(commanded(sim)) > 5  # a ramp, not one jump


def test_the_ramp_slows_after_the_first_touch():
    class Soft(Recording):
        """The thumb touches first at closure 0.3; the finger tips build force gradually from 0.5."""

        def read_fingertips(self, arm):
            c = self.hand_closure(arm)
            return {**dict.fromkeys(FINGER_TIPS, max(0.0, (c - 0.5) * 2.0)), "left_thumb_tip": 0.3 if c >= 0.3 else 0.0}

    sim, ctx = at_the_part(Soft())
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok
    steps = [b - a for a, b in zip(commanded(sim), commanded(sim)[1:], strict=False) if b > a]
    assert steps[0] == pytest.approx(1.0 * sim.dt * 2)  # fast before anything touches
    assert steps[-1] == pytest.approx(0.25 * sim.dt * 2)  # slow once a tip feels the part
    assert result.data["closure"] == pytest.approx(0.5 + 0.934 / 8.0, abs=0.006)  # stops as the four tips reach 0.93 N


def test_one_read_at_the_target_does_not_end_the_ramp():
    class OneSpike(Recording):
        reads = 0

        def read_fingertips(self, arm):
            self.reads += 1
            out = dict.fromkeys((*FINGER_TIPS, "left_thumb_tip"), 0.0)
            if self.reads == 3:  # one noisy read at the target, then nothing
                out["left_index_tip"] = 5.0
            return out

    sim, ctx = at_the_part(OneSpike())
    ctx.grip_timeout_s = 10.0
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.data["closure_cmd"] == pytest.approx(1.0)  # it kept closing after the spike


def test_closing_on_nothing_fails_as_no_touch():
    sim, ctx = at_the_part()
    sim.set_object_pose(ctx.target_part, (0.9, -0.3, ctx.layout.part_center_z), 0.0)  # the part is gone
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.reason == "hand closed without touching the part"
    assert result.data["control"] == "force" and result.data["closure_cmd"] == pytest.approx(1.0)
    assert result.data["touched"] is False


def test_force_reached_only_where_the_empty_hand_presses_on_itself_is_no_touch():
    class EmptyHand(Recording):
        """The index and thumb tips meet each other past the empty-hand closure, as measured in Genesis."""

        def read_fingertips(self, arm):
            pressed = 20.0 if self.hand_closure(arm) >= EMPTY_CLOSURE + 0.02 else 0.0
            return {**dict.fromkeys(FINGER_TIPS, 0.0), "left_index_tip": pressed, "left_thumb_tip": pressed}

    sim, ctx = at_the_part(EmptyHand())
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.reason.startswith("hand closed without touching the part (the fingers pressed on the hand itself")


def test_a_touch_that_never_builds_to_the_grip_force_fails():
    sim, ctx = at_the_part()
    sim.set_fingertips("left", dict.fromkeys(FINGER_TIPS, 0.21))  # four tips touching, 0.84 N together: short of 0.93
    ctx.grip_timeout_s = 10.0
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.reason.startswith("never reached grip force: 0.84 of 0.93 N on the finger tips with the hand fully closed")


def test_the_ramp_times_out():
    sim, ctx = at_the_part()
    ctx.grip_timeout_s = 0.1  # the fake's tips do not reach the part in 0.1 s at the fast rate
    sim.set_fingertips("left", {"left_thumb_tip": 0.5})  # something touches, so it is not a no-touch
    result = SKILLS["grasp"](sim, ctx)
    assert not result.ok
    assert result.reason.endswith("after 0.1 s")
    assert result.data["grasp_s"] == pytest.approx(0.1)


def test_a_set_grip_force_overrides_the_part_budget():
    sim, ctx = at_the_part()
    ctx.grip_force_n = 4.0
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok and result.data["target_n"] == 4.0 and result.data["finger_side_n"] >= 4.0


def test_the_old_fixed_closure_is_still_there():
    sim, ctx = at_the_part()
    ctx.grasp_control = "fixed"
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok
    assert commanded(sim)[-1] == pytest.approx(0.9) and len(commanded(sim)) == 2  # open at reach, then one jump to 0.9
    assert result.data["control"] == "fixed" and result.data["closure"] == pytest.approx(0.9) and result.data["contacts"] > 0


def test_a_hand_without_fingertip_sensors_falls_back_to_the_fixed_closure():
    sim, ctx = at_the_part(Recording(sensors=SensorConfig.off()))
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok
    assert result.data["control"] == "fixed (no fingertip sensors)"
    assert commanded(sim)[-1] == pytest.approx(0.9)


def test_an_unknown_grasp_control_is_refused():
    sim, ctx = at_the_part()
    ctx.grasp_control = "squeeze"
    with pytest.raises(ValueError, match="grasp_control"):
        SKILLS["grasp"](sim, ctx)


def test_weld_mode_welds_after_the_force_grip():
    sim, ctx = at_the_part()
    ctx.grasp_mode = "weld"
    result = SKILLS["grasp"](sim, ctx)
    assert result.ok and result.data["welded"] is True and result.data["control"] == "force"
    weld = sim.calls.index(f"attach:{ctx.target_part}")
    closes = [(i, float(c.rsplit(":", 1)[1])) for i, c in enumerate(sim.calls) if c.startswith("set_hand:left:")]
    before = [v for i, v in closes if i < weld]
    assert before  # the hand closed on the part before the weld
    # after it, the fingers only rest where they are (no squeeze into a part the weld holds still)
    assert all(v <= max(before) + 1e-9 for i, v in closes if i > weld)


def test_the_force_grip_carries_the_part_through_lift():
    sim, ctx = at_the_part()
    assert SKILLS["grasp"](sim, ctx).ok
    assert SKILLS["lift"](sim, ctx).ok


def test_the_grip_log_line_names_every_tip_the_force_and_the_outcome():
    sim, ctx = at_the_part()
    result = SKILLS["grasp"](sim, ctx)
    line = grip_line(ctx.target_part, result)
    assert line.startswith(f"grip {ctx.target_part}: force, closure 0.6")
    assert "of 0.93 N" in line and line.endswith("held")
    for tip in ("index", "middle", "ring", "little", "thumb"):
        assert f"{tip} " in line
    failed = grip_line("part_01", SkillResult(False, "hand closed without touching the part", {"control": "force", "target_n": 0.93}))
    assert failed.endswith("FAILED: hand closed without touching the part")


def test_the_grip_window_rows_are_panel_rows():
    tips = {"left_index_tip": 0.3, "left_middle_tip": 0.4, "left_ring_tip": 0.0, "left_little_tip": 20.0, "left_thumb_tip": 0.25}
    rows = grip_rows(tips, 0.93, 0.62)
    assert all(set(r) == {"label", "value", "detail", "state"} and r["state"] in ("ok", "warn", "idle") for r in rows)
    assert [r["label"] for r in rows] == ["finger side", "closure", "index", "middle", "ring", "little", "thumb"]
    assert rows[0]["value"] == f"{finger_side_n(tips, 'left'):.2f} N" and rows[0]["state"] == "ok"  # thumb not summed: 20.70
    assert rows[0]["value"] == "20.70 N"
    assert rows[4]["state"] == "idle" and rows[5]["state"] == "warn" and "20 N range" in rows[5]["detail"]
    assert grip_rows(dict.fromkeys(tips, 0.0), 0.93, 0.0)[0]["state"] == "idle"
    assert grip_rows({**dict.fromkeys(tips, 0.0), "left_index_tip": 0.5}, 0.93, 0.6)[0]["state"] == "warn"  # touching, short
