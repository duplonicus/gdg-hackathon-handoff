"""The two capsule cells as geometry (Nov 4): nothing clips, the tray fits its parchment, the robot can turn around."""

from __future__ import annotations

import numpy as np
import pytest

from handoff.capsule_cell import (
    MAX_LEAKERS,
    MAX_SPARES,
    CellLayout,
    TwoCells,
    cell_furniture,
    make_two_cell_scene,
    overlapping,
    park_for,
    visible,
)
from handoff.sim.base import BoxSpec


def test_overlap_check_finds_a_clip_and_ignores_touching_and_decals():
    a = BoxSpec("a", (0.2, 0.2, 0.1), (0.0, 0.0, 0.05), fixed=True)
    clip = BoxSpec("clip", (0.2, 0.2, 0.1), (0.19, 0.0, 0.05), fixed=True)  # 1 cm into a
    touch = BoxSpec("touch", (0.2, 0.2, 0.1), (0.0, 0.2, 0.05), fixed=True)  # face to face
    on_top = BoxSpec("on_top", (0.1, 0.1, 0.1), (0.0, 0.0, 0.15), fixed=True)  # resting on a
    decal = BoxSpec("decal", (0.1, 0.1, 0.01), (0.0, 0.0, 0.05), fixed=True, collision=False)
    found = overlapping([a, clip, touch, on_top, decal])
    assert [(x, y) for x, y, _ in found] == [("a", "clip")]
    assert found[0][2] == pytest.approx(0.01)


def test_overlap_check_turns_footprints_with_yaw():
    long_box = BoxSpec("long", (0.4, 0.02, 0.1), (0.0, 0.0, 0.05), yaw_deg=90.0, fixed=True)  # now runs along y
    beside = BoxSpec("beside", (0.1, 0.1, 0.1), (0.0, 0.15, 0.05), fixed=True)
    assert [(x, y) for x, y, _ in overlapping([long_box, beside])] == [("long", "beside")]


def test_nothing_in_either_cell_clips():
    """dup, Nov 4: the old trays clipped into each other on slide 1. Every visible solid in the built scene is checked."""
    scene = make_two_cell_scene()
    assert overlapping(visible(scene.boxes)) == []


def test_one_cell_alone_does_not_clip():
    assert overlapping(cell_furniture(CellLayout())) == []


def test_every_capsule_lies_on_the_parchment():
    lay = CellLayout()
    paper = next(b for b in cell_furniture(lay) if b.name == "parchment")
    half_x, half_y = paper.size[0] / 2.0, paper.size[1] / 2.0
    for x, y in lay.grid():
        assert abs(x - paper.pos[0]) + lay.capsule[0] / 2.0 <= half_x
        assert abs(y - paper.pos[1]) + lay.capsule[1] / 2.0 <= half_y
    assert len(lay.grid()) >= 30  # spaced for a pinch and inside the measured reach since Nov 5 (dup chose fingers over 260 packed)
    assert all(abs(y - lay.tray_center[1]) > lay.aisle / 2.0 for _, y in lay.grid())  # nothing in the aisle


def test_neighbouring_capsules_do_not_touch():
    lay = CellLayout()
    assert lay.pitch[0] > lay.capsule[0] and lay.pitch[1] > lay.capsule[1]


def test_good_bin_takes_a_whole_tray():
    lay = CellLayout()
    assert lay.good_bin_inner[0] >= max(lay.tray_size) and lay.good_bin_inner[1] >= min(lay.tray_size)


def test_robot_can_turn_around_at_its_spawn_in_the_aisle():
    """The robot spawns at x -1.2 (config) and turns in place between the cells: nothing solid within 0.5 m of it."""
    scene = make_two_cell_scene()
    sx, sy, _ = scene.robot_start
    for box in visible(scene.boxes):
        if box.pos[2] - box.size[2] / 2.0 > 1.8:
            continue  # above the robot
        dx = max(abs(box.pos[0] - sx) - box.size[0] / 2.0, 0.0)
        dy = max(abs(box.pos[1] - sy) - box.size[1] / 2.0, 0.0)
        assert float(np.hypot(dx, dy)) >= 0.5, box.name


def test_cells_face_each_other():
    a, b = TwoCells().cells
    ax, _ = a.to_world(0.7, 0.0)  # a point on each table
    bx, _ = b.to_world(0.7, 0.0)
    assert ax > -1.2 > bx


def test_scene_names_are_unique_and_hold_the_loose_leakers():
    scene = make_two_cell_scene()
    names = [b.name for b in scene.boxes]
    assert len(names) == len(set(names))
    loose = [b for b in scene.boxes if not b.fixed]
    assert sorted(b.name for b in loose) == sorted(TwoCells().leaker_names + TwoCells().spare_names)
    assert len(loose) == MAX_LEAKERS + MAX_SPARES


def test_each_leaker_bin_sits_beside_its_half_of_the_tray():
    """The station's bins: one per hand, on the table, clear of the tray, within the reach probe_station.py measured
    (bins at x 0.25 to 0.40, |y| 0.30 to 0.36, reached by the hand on that side)."""
    lay = CellLayout()
    tray_edge = lay.tray_size[1] / 2.0
    for side, sign in (("left", 1.0), ("right", -1.0)):
        bx, by = lay.leaker_bin(side)
        assert sign * by > tray_edge + lay.leaker_bin_inner[1] / 2.0 + lay.bin_thick  # clear of the tray
        assert 0.25 <= bx <= 0.40 and 0.30 <= abs(by) <= 0.36  # the bin spots the station probe reached


def test_park_for_refuses_a_spot_past_the_legs_or_out_of_reach():
    lay = CellLayout()
    with pytest.raises(ValueError):
        park_for(lay, 0.35, -0.85)  # the base would sit outside the right legs
    with pytest.raises(ValueError):
        park_for(lay, 0.70, 0.0)  # deeper than the hand reaches from the closest spot


def test_each_tray_is_lit_by_two_lamps_and_no_sun():
    """dup (Nov 6): the kit's sun threw the robot's arm across the tray in shade bands that drove d3's false alarms.
    Each tray has a lamp over each half; the scene has no sun."""
    cells = TwoCells()
    scene = make_two_cell_scene(cells)
    assert scene.sun == 0.0
    lay = cells.layout
    for cell in cells.cells:
        over = [l for l in scene.lamps if np.hypot(*(np.array(l[:2]) - np.array(cell.to_world(*lay.tray_center)))) < 0.3]
        assert len(over) == 2 and all(l[2] > lay.table_top for l in over)


def test_whatever_waits_under_the_floor_is_hidden_by_a_drawn_floor_above_it():
    """The kit's plane is not drawn in capsule scenes; every box parked under the floor (the park 40 m aside) must still
    have a drawn floor slab over it, or a viewer turned that way sees it hanging in the void."""
    from handoff.capsule_cell import TwoCells, make_two_cell_scene

    scene = make_two_cell_scene(TwoCells())
    assert scene.kit_floor is False
    slabs = [b for b in scene.boxes if b.name.startswith("floor_")]
    for box in scene.boxes:
        top = box.pos[2] + box.size[2] / 2.0
        if top >= 0.0 or box in slabs:
            continue
        over = [s for s in slabs if abs(box.pos[0] - s.pos[0]) <= s.size[0] / 2.0 and abs(box.pos[1] - s.pos[1]) <= s.size[1] / 2.0 and s.pos[2] - s.size[2] / 2.0 > top]
        assert over, f"{box.name} at {box.pos} has no drawn floor over it"
