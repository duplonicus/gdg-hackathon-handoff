"""Removing a leaker on the fake simulator: from the station, the hand on its side picks it and drops it in that side's bin."""

from __future__ import annotations

import numpy as np
import pytest

from handoff.capsule_cell import TwoCells, make_two_cell_scene
from handoff.capsule_skills import park_station, remove_leaker
from handoff.sim.fake import FakeSimulator


def place_leaker(sim, cells, cell_name, spot):
    """Swap grid capsule ``spot`` for loose leaker 0 (and its pool), as an episode does."""
    cell, lay = cells.cell(cell_name), cells.layout
    x, y = cell.to_world(*lay.grid()[spot])
    sim.set_object_pose(cells.grid_names[spot], (x, y, -2.0))
    sim.set_object_pose(cells.leaker_names[0], (x, y, lay.paper_top + lay.capsule[2] / 2.0), cell.yaw_deg)
    sim.set_object_pose(cells.pool_names[0], (x, y, lay.paper_top + 0.0003))


def in_leaker_bin(cells, cell_name, pos, side) -> bool:
    cell, lay = cells.cell(cell_name), cells.layout
    cx, cy = cell.to_world(*lay.leaker_bin(side))
    half = lay.leaker_bin_inner[0] / 2.0
    floor = lay.table_top + lay.leaker_bin_floor
    rim = floor + lay.leaker_bin_wall
    # resting anywhere from the floor up: a finished cell's bin already holds a heap of leakers
    return abs(pos[0] - cx) <= half and abs(pos[1] - cy) <= half and floor - 1e-6 <= pos[2] - lay.capsule[2] / 2.0 <= rim


@pytest.mark.parametrize("cell_name", ["a", "b"])
@pytest.mark.parametrize("spot", [0, 3, 13, 16, 31])  # near-right outer, far-right outer, right inner, left inner, far-left outer (4 rows x 8)
def test_a_leaker_ends_up_in_the_bin_on_its_side_without_driving(cell_name, spot):
    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, cell_name, spot)
    assert park_station(sim, cells.cell(cell_name)).ok
    base = sim.base_pose().pos.copy()
    result = remove_leaker(sim, cells.cell(cell_name), cells.leaker_names[0])
    assert result.ok, result.reason
    assert np.allclose(sim.base_pose().pos, base)  # only the arm moved
    pos = sim.object_state(cells.leaker_names[0]).pos
    side = "left" if cells.layout.grid()[spot][1] >= 0 else "right"
    assert in_leaker_bin(cells, cell_name, pos, side), pos


def test_the_pinch_never_goes_below_the_capsules_centre():
    """The neighbours are fixed, so the fake cannot show a bump; what it can show is every wrist target: none puts the
    pinch point below the capsule's centre (palm-down pinch, Nov 5)."""
    from handoff.capsule_skills import PINCH

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 30)
    park_station(sim, cells.cell("a"))
    remove_leaker(sim, cells.cell("a"), cells.leaker_names[0])
    lay = cells.layout
    lowest = lay.paper_top + lay.capsule[2] / 2.0 - PINCH[2]
    assert sim.moves and min(float(m.target[2]) for m in sim.moves) >= lowest - 1e-9


def test_a_failed_pick_lets_go_of_the_capsule(monkeypatch):
    import handoff.capsule_skills as skills
    from handoff.capsule_skills import MoveResult

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 30)
    park_station(sim, cells.cell("a"))

    def pick(sim_, cell, name, top_z, arm="left"):
        sim_.attach(name, arm)  # welded, then the lift check fails
        return MoveResult(False, "capsule rose only 2 mm", 0.0)

    monkeypatch.setattr(skills, "pick_capsule", pick)
    result = skills.remove_leaker(sim, cells.cell("a"), cells.leaker_names[0])
    assert not result.ok
    assert cells.leaker_names[0] not in sim._welded


def test_a_hand_that_misses_straight_ahead_turns_toward_the_middle(monkeypatch):
    """The station's reach: when the straight-ahead heading misses at the lift height, the next heading is tried."""
    from handoff.capsule_skills import hand_quat, reach_over
    from handoff.sim.base import yaw_of

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    park_station(sim, cells.cell("a"))
    real = sim.move_wrist
    straight = hand_quat(yaw_of(sim.base_pose().quat), "left", 0.0)

    def move(arm, pos, quat, **kw):
        if np.allclose(quat, straight) or np.allclose(quat, -straight):
            real(arm, pos, quat, **kw)
            return 0.05  # straight ahead falls short
        return real(arm, pos, quat, **kw)

    monkeypatch.setattr(sim, "move_wrist", move)
    lay = cells.layout
    err, heading, _ = reach_over(sim, "left", cells.cell("a").to_world(*lay.grid()[30]), lay.paper_top + lay.capsule[2], 0.004, 0.005)
    assert heading == 30.0 and err <= 0.005


def test_when_no_heading_is_within_tolerance_the_hand_ends_at_the_best_ones_pick_height(monkeypatch):
    """Review of 72a8c9f: the hand ended at LIFT on the last heading while the error of an earlier one was returned,
    so a 5 to 15 mm near miss welded the capsule from 10 cm up."""
    from handoff.capsule_skills import PINCH, hand_quat, reach_over
    from handoff.sim.base import yaw_of

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    park_station(sim, cells.cell("a"))
    real = sim.move_wrist
    lay = cells.layout
    top = lay.paper_top + lay.capsule[2]
    yaw = yaw_of(sim.base_pose().quat)
    thirty = hand_quat(yaw, "left", 30.0)

    def move(arm, pos, quat, **kw):
        real(arm, pos, quat, **kw)
        if pos[2] < top - PINCH[2] + 0.05:  # at pick height (not LIFT): heading 30 misses by 8 mm, every other by 12
            return 0.008 if (np.allclose(quat, thirty) or np.allclose(quat, -thirty)) else 0.012
        return 0.0

    monkeypatch.setattr(sim, "move_wrist", move)
    err, heading, _ = reach_over(sim, "left", cells.cell("a").to_world(*lay.grid()[30]), top, 0.004, 0.005)
    assert (err, heading) == (0.008, 30.0)
    assert float(sim._wrist["left"].pos[2]) == pytest.approx(top + 0.004 - PINCH[2])


def test_a_wrist_move_repeats_while_it_still_closes_in_and_stops_when_it_stalls():
    """_settle_on (0ebefd4): from the forearms-up pose one move fell short and repeats closed in (51, 42, 31 mm)."""
    from handoff.capsule_skills import _settle_on

    class Arm:
        def __init__(self, errors):
            self.errors, self.calls = list(errors), 0

        def move_wrist(self, *a, **k):
            self.calls += 1
            return self.errors.pop(0)

    closing = Arm([0.051, 0.042, 0.031, 0.020])
    assert _settle_on(closing, "left", np.zeros(3), np.zeros(4)) == 0.020 and closing.calls == 4  # every try gained > 5 mm
    stalled = Arm([0.040, 0.038, 0.001])
    assert _settle_on(stalled, "left", np.zeros(3), np.zeros(4)) == 0.038 and stalled.calls == 2  # gained 2 mm: stop
    there = Arm([0.004])
    assert _settle_on(there, "left", np.zeros(3), np.zeros(4)) == 0.004 and there.calls == 1


def test_a_low_hand_goes_straight_up_before_the_arms_fold():
    """Folding to the travel pose from 4 mm over the tray swept the hand through the capsules (review of 0ebefd4)."""
    from handoff.capsule_skills import CLEAR_Z, travel_pose

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    park_station(sim, cells.cell("a"))
    lay = cells.layout
    low = np.array([*cells.cell("a").to_world(0.35, 0.10), lay.paper_top + 0.05])
    sim.move_wrist("left", low, sim.wrist_pose("left").quat)
    sim.moves.clear()
    travel_pose(sim)
    up = [m for m in sim.moves if m.arm == "left"]
    assert up and np.allclose(up[0].target[:2], low[:2]) and up[0].target[2] == pytest.approx(CLEAR_Z)


def test_the_pick_only_pinches_with_the_hand_straight_ahead(monkeypatch):
    """Review of fdfd388: turned, the open tips land on the next column, so a pick that misses straight ahead fails
    rather than turning the hand."""
    from handoff.capsule_skills import pick_capsule

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 16)
    park_station(sim, cells.cell("a"))
    real = sim.move_wrist
    quats = []

    def move(arm, pos, quat, **kw):
        quats.append(np.asarray(quat, dtype=float))
        real(arm, pos, quat, **kw)
        return 0.05  # nothing reaches

    monkeypatch.setattr(sim, "move_wrist", move)
    lay = cells.layout
    result = pick_capsule(sim, cells.cell("a"), cells.leaker_names[0], lay.paper_top + lay.capsule[2], "left")
    assert not result.ok and cells.leaker_names[0] not in sim._welded
    assert all(np.allclose(q, quats[0]) or np.allclose(q, -quats[0]) for q in quats)  # one heading only


def test_the_carry_goes_straight_up_before_it_crosses_to_the_bin():
    """dup, watching the viewer (Nov 5): the hand hit the bin every time it rose from the pick toward it. The move after
    the pick's lift is vertical, to the carry height over the rim, and only then across."""
    from handoff.capsule_skills import LIFT, wrist_for
    from handoff.sim.base import yaw_of

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 16)
    park_station(sim, cells.cell("a"))
    sim.moves.clear()
    assert remove_leaker(sim, cells.cell("a"), cells.leaker_names[0]).ok
    moves = [m for m in sim.moves if m.arm == "left"]
    lay = cells.layout
    rim = lay.table_top + lay.leaker_bin_floor + lay.leaker_bin_wall
    carry_z = float(wrist_for(cells.cell("a").to_world(*lay.leaker_bin("left")), rim, yaw_of(sim.base_pose().quat), LIFT, "left")[2])
    rise = next(i for i, m in enumerate(moves) if m.target[2] == pytest.approx(carry_z) and np.allclose(m.target[:2], m.start[:2]))
    assert all(np.allclose(m.target[:2], m.start[:2]) or m.target[2] >= carry_z - 1e-6 for m in moves[rise:])  # nothing sideways below it


def test_a_single_heading_reach_that_lands_a_little_off_does_not_go_up_and_down_again(monkeypatch):
    """dup (Nov 6): the arm went down, up, down, then grabbed: with one heading, a descent 5 to 15 mm off backed up and
    came down again on the same heading. It now stays down and reports that error."""
    from handoff.capsule_skills import PICK_HEADINGS, reach_over

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    park_station(sim, cells.cell("a"))
    real = sim.move_wrist
    lay = cells.layout
    centre = lay.paper_top + lay.capsule[2] / 2.0
    downs = []

    def move(arm, pos, quat, **kw):
        real(arm, pos, quat, **kw)
        if pos[2] < centre + 0.08:  # a descent to the pick
            downs.append(pos.copy())
            return 0.008
        return 0.0

    monkeypatch.setattr(sim, "move_wrist", move)
    err, heading, _ = reach_over(sim, "left", cells.cell("a").to_world(*lay.grid()[16]), centre, 0.0, 0.005, PICK_HEADINGS)
    assert (err, heading) == (0.008, 0.0)
    assert len(downs) == 1  # one descent, no second


def test_the_head_watches_the_tray_the_capsule_and_the_bin(monkeypatch):
    """dup (Nov 6): "its head isn't looking down at the table". Parked, the head looks down at the tray; it follows the
    pick and then the bin (the pictures still come from the fixed cameras)."""
    import handoff.capsule_skills as skills

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 16)
    monkeypatch.setattr(skills, "LOOK_AT_WORK", True)  # viewer runs only
    heads = []
    real = sim.set_head
    sim.set_head = lambda yaw, pitch: (heads.append((yaw, pitch)), real(yaw, pitch))[1]
    assert park_station(sim, cells.cell("a")).ok
    assert heads and heads[-1][1] > 0.5  # looking well down at the tray
    remove_leaker(sim, cells.cell("a"), cells.leaker_names[0])
    assert len(heads) >= 3  # tray, capsule, bin
    left = [y for y, _ in heads[1:]]
    assert any(y < -0.3 for y in left)  # turned to the left bin (head_yaw negative turns it left)


def test_data_runs_keep_the_head_level():
    """The frozen pictures (sel02, test05) are made with the head level: off by default."""
    import handoff.capsule_skills as skills

    assert skills.LOOK_AT_WORK is False
    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 16)
    park_station(sim, cells.cell("a"))
    remove_leaker(sim, cells.cell("a"), cells.leaker_names[0])
    assert sim._head == (0.0, 0.0)


def test_the_drop_goes_inside_the_bin_and_opens_before_letting_go():
    """sel02 (Nov 6): released 2 cm above the rim, 20 of 299 leakers ended outside the bins. The pinch now goes below
    the rim, and the fingers open while the capsule is still held, so nothing flicks it as it falls."""
    from handoff.capsule_skills import DROP_INTO_BIN, PINCH

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_leaker(sim, cells, "a", 16)
    park_station(sim, cells.cell("a"))
    sim.calls.clear()
    sim.moves.clear()
    assert remove_leaker(sim, cells.cell("a"), cells.leaker_names[0]).ok
    lay = cells.layout
    rim = lay.table_top + lay.leaker_bin_floor + lay.leaker_bin_wall
    from handoff.capsule_skills import wrist_for
    from handoff.sim.base import yaw_of

    over_bin = wrist_for(cells.cell("a").to_world(*lay.leaker_bin("left")), rim, yaw_of(sim.base_pose().quat), 0.0, "left")
    at_bin = [float(m.target[2]) for m in sim.moves if m.arm == "left" and np.hypot(*(m.target[:2] - over_bin[:2])) < 0.02]
    assert min(at_bin) == pytest.approx(rim - DROP_INTO_BIN - PINCH[2])  # the pinch below the rim at the drop
    detach = sim.calls.index(f"detach:{cells.leaker_names[0]}")
    assert any(c.startswith("attach") for c in sim.calls[:detach])
