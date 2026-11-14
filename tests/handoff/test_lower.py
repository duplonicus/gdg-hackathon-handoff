"""The part is set down into the bin, not dropped from the hover: the wrist comes down to a measured height above the wall."""

import numpy as np

from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext


def carried_to_hover(seed: int = 0, **ctx_kw) -> tuple[FakeSimulator, TaskContext]:
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, **ctx_kw)
    for name in ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift", "move_to_bin"):
        assert SKILLS[name](sim, ctx).ok, name
    return sim, ctx


def test_lower_brings_the_wrist_straight_down_to_the_release_height_with_the_part_still_held():
    sim, ctx = carried_to_hover()
    region = ctx.layout.bin_region(ctx.target_bin)
    hover = sim.wrist_pose(ctx.arm).pos.copy()
    assert hover[2] == region.wall_top_z + ctx.hover_above_bin
    result = SKILLS["lower"](sim, ctx)
    assert result.ok, result.reason
    wrist = sim.wrist_pose(ctx.arm).pos
    assert np.allclose(wrist[:2], hover[:2])  # straight down, no drift across the bin
    assert abs(wrist[2] - (region.wall_top_z + ctx.release_above_wall)) < 1e-9
    assert result.data["held"]
    assert abs(result.data["wrist_above_wall"] - ctx.release_above_wall) < 1e-9
    # The hand cannot enter the bin (19 cm long, 16 cm inside), so the release height stays above the wall top.
    assert ctx.release_above_wall > 0.0


def test_lower_cuts_the_drop_to_the_bin_floor():
    sim, ctx = carried_to_hover()
    region = ctx.layout.bin_region(ctx.target_bin)
    half_h = ctx.layout.part_size[2] / 2.0
    drop_from_hover = sim.object_state(ctx.target_part).pos[2] - half_h - region.floor_z
    SKILLS["lower"](sim, ctx)
    drop_after = sim.object_state(ctx.target_part).pos[2] - half_h - region.floor_z
    assert abs((drop_from_hover - drop_after) - (ctx.hover_above_bin - ctx.release_above_wall)) < 1e-9
    assert drop_after < 0.08  # the measured hover drop was 11 cm


def test_lower_reports_a_slipped_part():
    sim, ctx = carried_to_hover()
    sim.set_hand(ctx.arm, 0.0)  # the part falls out before the descent and ends up back on the table
    sim.set_object_pose(ctx.target_part, (0.9, -0.3, ctx.layout.part_center_z), 0.0)
    sim.set_hand(ctx.arm, ctx.close_fraction)  # the hand closes on nothing
    result = SKILLS["lower"](sim, ctx)
    assert not result.ok
    assert "slipped" in result.reason
