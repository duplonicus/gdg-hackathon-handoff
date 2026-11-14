"""The success checks on synthetic world states (no simulator, no GPU)."""

import numpy as np

from handoff.checks import WorldState, any_on_floor, grasp_held, in_bin, lifted, moved_with_hand, on_floor, still
from handoff.scene import Layout
from handoff.sim.base import ObjectState, Pose

LAYOUT = Layout()
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def state(part_pos, wrist_pos=(0.4, 0.3, 0.9), contacts=0, vel=(0.0, 0.0, 0.0), t=0.0) -> WorldState:
    return WorldState(
        t=t,
        objects={"part_0": ObjectState(np.array(part_pos, dtype=float), IDENTITY, np.array(vel, dtype=float))},
        wrists={"left": Pose(np.array(wrist_pos, dtype=float), IDENTITY)},
        hand_closure={"left": 0.9},
        contacts={("left", "part_0"): contacts},
        table_top=LAYOUT.table_top,
        bins=LAYOUT.bin_regions(),
        part_size=LAYOUT.part_size,
    )


def test_part_on_table_is_not_lifted():
    assert not lifted(state(LAYOUT.slot_pos(0)), "part_0")


def test_grasp_held_needs_contact_proximity_and_lift():
    z = LAYOUT.part_center_z + 0.10
    held = state((0.5, 0.26, z), wrist_pos=(0.42, 0.30, z + 0.01), contacts=3)
    assert grasp_held(held, "part_0")
    assert not grasp_held(state((0.5, 0.26, z), wrist_pos=(0.42, 0.30, z), contacts=0), "part_0")
    assert not grasp_held(state((0.5, 0.26, z), wrist_pos=(0.9, 0.30, z), contacts=3), "part_0")
    assert not grasp_held(state((0.5, 0.26, LAYOUT.part_center_z), wrist_pos=(0.42, 0.30, 0.87), contacts=3), "part_0")


def test_moved_with_hand_compares_displacements():
    before = state((0.5, 0.26, 0.86), wrist_pos=(0.42, 0.30, 0.87))
    after_good = state((0.5, 0.26, 1.01), wrist_pos=(0.42, 0.30, 1.02))
    after_slipped = state((0.5, 0.26, 0.86), wrist_pos=(0.42, 0.30, 1.02))
    assert moved_with_hand(before, after_good, "part_0")
    assert not moved_with_hand(before, after_slipped, "part_0")


def test_in_bin_requires_footprint_floor_and_rest():
    region = LAYOUT.bin_region("good")
    resting_z = region.floor_z + LAYOUT.part_size[2] / 2.0
    assert in_bin(state((region.cx, region.cy, resting_z)), "part_0", "good")
    # lying on its side counts too
    assert in_bin(state((region.cx + 0.05, region.cy - 0.05, region.floor_z + LAYOUT.part_size[0] / 2.0)), "part_0", "good")
    assert not in_bin(state((region.cx + 0.15, region.cy, resting_z)), "part_0", "good")  # outside the footprint
    assert in_bin(state((region.cx, region.cy, resting_z + LAYOUT.part_size[2])), "part_0", "good")  # stacked on another part
    assert not in_bin(state((region.cx, region.cy, resting_z + 0.2)), "part_0", "good")  # hovering above
    assert in_bin(state((region.cx, region.cy, resting_z), vel=(0.0, 0.0, -0.5)), "part_0", "good")  # reported speed is ignored
    assert not in_bin(state((region.cx, region.cy, resting_z), vel=(0.0, 0.0, -0.5)), "part_0", "good", max_speed=0.1)  # unless asked
    assert not in_bin(state((region.cx, region.cy, resting_z)), "part_0", "rework")


def test_on_floor():
    assert on_floor(state((0.5, 0.25, 0.04)), "part_0")
    assert not on_floor(state(LAYOUT.slot_pos(0)), "part_0")
    assert any_on_floor(state((0.5, 0.25, 0.04))) == ["part_0"]


def test_still_compares_positions_over_time():
    a = state((0.5, 0.4, 0.85), vel=(0.3, 0.0, 0.0))  # a jittery velocity readout
    b = state((0.5, 0.4, 0.85), vel=(0.3, 0.0, 0.0))
    assert still(a, b, "part_0")
    assert not still(a, state((0.52, 0.4, 0.85)), "part_0")
