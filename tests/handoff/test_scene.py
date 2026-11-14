"""Scene geometry and jitter (no simulator)."""

import numpy as np

from handoff.scene import Layout, make_episode, make_scene
from handoff.sim.fake import REACH_M, SHOULDER
from handoff.skills import TaskContext

LAYOUT = Layout()


def test_parts_and_bins_sit_on_the_table():
    scene = make_scene(LAYOUT)
    table = scene.box("table")
    assert abs(table.pos[2] + table.size[2] / 2.0 - LAYOUT.table_top) < 1e-9
    (x0, x1), (y0, y1) = LAYOUT.table_x, LAYOUT.table_y
    for box in scene.boxes:
        if box.name == "table" or not box.fixed:
            continue  # parts start parked off the table; episodes place them
        hx, hy = box.size[0] / 2.0, box.size[1] / 2.0
        assert x0 <= box.pos[0] - hx and box.pos[0] + hx <= x1, box.name
        assert y0 <= box.pos[1] - hy and box.pos[1] + hy <= y1, box.name
        assert box.pos[2] - box.size[2] / 2.0 >= LAYOUT.table_top - 1e-9, box.name


def test_slots_do_not_overlap_bins_or_each_other():
    scene = make_scene(LAYOUT)
    fixed = [b for b in scene.boxes if b.fixed and b.name != "table"]
    slots = [LAYOUT.slot_pos(i) for i in range(len(LAYOUT.slot_xy))]
    for i, slot in enumerate(slots):
        for other in fixed:
            gap_x = abs(slot[0] - other.pos[0]) - (LAYOUT.part_size[0] + other.size[0]) / 2.0
            gap_y = abs(slot[1] - other.pos[1]) - (LAYOUT.part_size[1] + other.size[1]) / 2.0
            assert max(gap_x, gap_y) > 0.02, (i, other.name)
        for j, other_slot in enumerate(slots):
            if j != i:
                assert abs(slot[0] - other_slot[0]) + abs(slot[1] - other_slot[1]) > 0.1


def test_pool_has_every_finish_and_parts_start_parked_off_the_table():
    scene = make_scene(LAYOUT)
    parts = [b for b in scene.boxes if not b.fixed]
    assert len(parts) == 3 * LAYOUT.pool_per_kind
    kinds = {b.finish.split("@")[0] for b in parts}
    assert kinds == {"good", "scratched", "unpainted"}
    assert all(b.texture is not None and b.texture.shape == (128, 128, 3) for b in parts)
    assert all(b.pos[0] > LAYOUT.table_x[1] for b in parts)


def test_episode_is_deterministic_and_jitter_is_bounded():
    a = make_episode(LAYOUT, 3)
    b = make_episode(LAYOUT, 3)
    c = make_episode(LAYOUT, 4)
    assert a == b
    assert a != c
    assert len(set(a.parts)) == len(LAYOUT.slot_xy)
    for slot, name in enumerate(a.parts):
        pos, yaw = a.poses[name]
        base = LAYOUT.slot_pos(slot)
        assert abs(pos[0] - base[0]) <= LAYOUT.jitter_xy_m
        assert abs(pos[1] - base[1]) <= LAYOUT.jitter_xy_m
        assert pos[2] == base[2]
        assert abs(yaw) <= LAYOUT.jitter_yaw_deg
        assert a.truth(name).kind in ("good", "scratched", "unpainted")


def test_grasp_and_hover_points_are_inside_measured_reach():
    ctx = TaskContext(layout=LAYOUT)
    part = np.array(LAYOUT.slot_pos(0))
    grasp = part + np.array(ctx.grasp_offset)
    assert np.linalg.norm(grasp - SHOULDER["left"]) < REACH_M - 0.03
    for name in LAYOUT.bin_names:
        region = LAYOUT.bin_region(name)
        hover = np.array([region.cx + ctx.grasp_offset[0], region.cy + ctx.grasp_offset[1], region.wall_top_z + ctx.hover_above_bin])
        assert np.linalg.norm(hover - SHOULDER["left"]) < REACH_M - 0.01, name
        assert 0.15 <= hover[1] <= 0.62, name  # the swept band where the left arm reaches with the grasp orientation
