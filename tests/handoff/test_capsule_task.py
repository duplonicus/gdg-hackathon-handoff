"""The capsule shift on the fake simulator (Nov 4): the tree, the questions it asks, and what ends up where."""

from __future__ import annotations

import numpy as np
import pytest

from handoff.bt import Status
from handoff.capsule_cell import TwoCells, make_two_cell_scene
from handoff.capsule_episode import CapsuleEpisode, make_capsule_episode, place_capsule_episode
from handoff.capsule_perception import SeenCapsule
from handoff.decider import Ladder, Step
from handoff.rungs import OracleRung
from handoff.sim.fake import FakeSimulator
from handoff.tasks.capsules import CapsuleContext, blocks, run_capsules


def truth_see(sim, ctx):
    """A stand-in for the tray camera: every capsule lying on the work cell's parchment, where the simulator has it."""
    if ctx.episode.work_cell is None:
        return np.zeros((PX, PX, 3), dtype=np.uint8), []
    lay = ctx.cells.layout
    names = list(ctx.cells.grid_names) + list(ctx.cells.leaker_names) + list(ctx.cells.spare_names)
    caps = []
    for name in names:
        pos = sim.object_state(name).pos
        if abs(float(pos[2]) - (lay.paper_top + lay.capsule[2] / 2.0)) <= 0.005:
            caps.append(SeenCapsule((float(pos[0]), float(pos[1]), float(pos[2])), pixel_box(pos, lay), 72))
    return np.zeros((PX, PX, 3), dtype=np.uint8), caps


PX = 2400  # a top-down picture of the aisle at 2 mm a pixel: rows along world x, columns along world y, from (-2.4, -2.4)


def pixel_box(pos, lay):
    """Where a capsule lying along world x shows in that picture: 12 by 6 pixels (24 by 11 mm)."""
    r = int(round((float(pos[0]) + 2.4) * 500))
    c = int(round((float(pos[1]) + 2.4) * 500))
    half_r, half_c = int(round(lay.capsule[0] * 250)), int(round(lay.capsule[1] * 250))
    return (r - half_r, c - half_c, r + half_r, c + half_c)


def run(episode: CapsuleEpisode, answer=None):
    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_capsule_episode(sim, cells, episode)
    holder = {}
    rung = OracleRung(answer or (lambda q: holder["ctx"].truth_now))
    ctx = CapsuleContext(cells=cells, episode=episode, ladder=Ladder([Step(rung, 0.5)]), see=truth_see)
    holder["ctx"] = ctx
    return sim, ctx, run_capsules(sim, ctx)


def with_leakers(n: int, cell: str) -> CapsuleEpisode:
    return next(e for s in range(500) if (e := make_capsule_episode(TwoCells(), s)).work_cell == cell and len(e.leakers) == n)


@pytest.mark.parametrize("cell", ["a", "b"])
def test_with_a_perfect_inspector_every_leaker_ends_in_the_bin(cell):
    episode = with_leakers(3, cell)
    sim, ctx, result = run(episode)
    assert result.status is Status.SUCCESS, result.trace.text()
    assert result.checks["went_to"] == cell
    assert result.checks["leakers_left_on_tray"] == 0
    assert result.checks["leakers_in_bin"] == 3
    assert result.checks["good_capsules_thrown_out"] == 0
    asked = [d.question_id for d in result.decisions]
    assert asked[0] == "where_needed" and asked[-1] == "tray_clear"
    assert asked.count("leaker_landed") == 3
    # an unclear picture (a leaker only partly in view, truth None) is left out of scoring since e5ef177
    assert all(d.answer == d.truth for d in result.decisions if d.truth is not None)


def test_nowhere_to_go_asks_once_and_stays():
    sim, ctx, result = run(CapsuleEpisode(1, None, (), ()))
    assert result.status is Status.SUCCESS
    assert [d.question_id for d in result.decisions] == ["where_needed"]
    assert result.checks["went_to"] is None


def test_a_clean_tray_is_scanned_and_left_alone():
    sim, ctx, result = run(with_leakers(0, "a"))
    assert result.status is Status.SUCCESS
    asked = [d.question_id for d in result.decisions]
    assert "leak_capsule" not in asked and asked.count("leak_block") == len(blocks(truth_see(sim, ctx)[1], ctx))
    assert ctx.removed == []


def test_a_false_alarm_throws_out_a_good_capsule_and_says_so():
    """An inspector that calls every capsule of a flagged block a leaker: the good ones it picks are spares."""
    episode = with_leakers(1, "a")
    holder = {}

    def jumpy(q):
        if q.id == "leak_capsule":
            return "leaker"
        return holder["ctx"].truth_now

    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_capsule_episode(sim, cells, episode)
    ctx = CapsuleContext(cells=cells, episode=episode, ladder=Ladder([Step(OracleRung(jumpy), 0.5)]), see=truth_see)
    holder["ctx"] = ctx
    result = run_capsules(sim, ctx)
    assert result.checks["leakers_left_on_tray"] == 0
    assert result.checks["good_capsules_thrown_out"] == len(cells.spare_names)  # it ran out of spares: the rest stay put
    assert result.checks["leakers_in_bin"] == 1


def test_blocks_cover_every_capsule_once_and_hold_at_most_the_block_size():
    episode = with_leakers(2, "b")
    cells = TwoCells()
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_capsule_episode(sim, cells, episode)
    for size in (1, 4, 16, 64):
        ctx = CapsuleContext(cells=cells, episode=episode, ladder=Ladder([]), block=size, cell="b")
        caps = truth_see(sim, ctx)[1]
        groups = blocks(caps, ctx)
        flat = [c for g in groups for c in g]
        assert sorted(map(id, flat)) == sorted(map(id, caps))
        assert max(len(g) for g in groups) <= size


def test_a_block_picture_that_shows_a_neighbours_leaker_is_labelled_yes():
    """Commit review round 4: the pad around a block shows the next block's edge capsules; a leaker among them makes
    the picture's truth yes, whichever block it belongs to."""
    cells = TwoCells()
    rows = len({round(x, 6) for x, _ in cells.layout.grid()})  # capsules along x per column (the grid is column by column)
    spot = 2 * rows  # column 2 of the right half: the first column of the second 2 x 2 block, beside the first block
    episode = CapsuleEpisode(3, "a", (spot,), (2,))
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_capsule_episode(sim, cells, episode)
    holder = {}
    ctx = CapsuleContext(cells=cells, episode=episode, ladder=Ladder([Step(OracleRung(lambda q: holder["ctx"].truth_now), 0.5)]), see=truth_see, cell="a", block=4)
    holder["ctx"] = ctx
    _, caps = truth_see(sim, ctx)  # the order the scan will number them in, before the leaker leaves the tray
    want = cells.cell("a").to_world(*cells.layout.grid()[spot])
    index = int(np.argmin([np.hypot(c.xyz[0] - want[0], c.xyz[1] - want[1]) for c in caps]))
    ctx.cell = None
    result = run_capsules(sim, ctx)
    asked = [d for d in result.decisions if d.question_id == "leak_block"]
    first_scan = asked[: len(blocks(caps, CapsuleContext(cells=cells, episode=episode, ladder=Ladder([]), cell="a", block=4)))]
    showing = [d for d in first_scan if index in d.context["in_view"]]
    assert len(showing) >= 2  # its own block and the one beside it
    own = [d for d in showing if index in d.context["capsules"]]
    assert len(own) == 1 and own[0].truth == "yes"  # wholly in its own block's picture
    assert all(d.truth in ("yes", None) for d in showing)  # in part beside it: never a clean "no"
    assert all(d.truth == "no" for d in first_scan if d not in showing)


def test_a_failed_drive_still_backs_out_and_reports_failure(monkeypatch):
    import handoff.tasks.capsules as task
    from handoff.capsule_skills import MoveResult

    real = task.park_station
    monkeypatch.setattr(task, "park_station", lambda *a, **k: (real(*a, **k), MoveResult(False, "parked 90 mm off", 0.09))[1])
    reversed_ = []
    real_back = task.back_out
    monkeypatch.setattr(task, "back_out", lambda sim: (reversed_.append(1), real_back(sim))[1])
    sim, ctx, result = run(with_leakers(1, "a"))
    assert result.status is Status.FAILURE
    assert reversed_ == [1]


def test_a_leaker_that_settled_toward_a_neighbour_labels_one_capsule_not_two():
    """Smoke run, Nov 4: a leaker 4 mm off its spot sat 12 mm from a neighbour, and both were labelled leakers."""
    from handoff.tasks.capsules import leaker_owners

    cells = TwoCells()
    episode = CapsuleEpisode(5, "a", (30,), (1,))
    sim = FakeSimulator()
    sim.build(make_two_cell_scene(cells))
    place_capsule_episode(sim, cells, episode)
    leaker = cells.grid_names[30]  # a leaker is the tray capsule at its spot
    pos = np.array(sim.object_state(leaker).pos)
    sim.set_object_pose(leaker, (float(pos[0]), float(pos[1]) + 0.004, float(pos[2])), 0.0)  # 4 mm toward the next row
    ctx = CapsuleContext(cells=cells, episode=episode, ladder=Ladder([]), see=truth_see, cell="a")
    _, caps = truth_see(sim, ctx)
    owners = leaker_owners(sim, ctx, caps)
    assert list(owners.values()) == [leaker]
    (index,) = owners
    assert np.hypot(caps[index].xyz[0] - pos[0], caps[index].xyz[1] - pos[1] - 0.004) < 1e-6  # the leaker itself, not its neighbour


def test_block_truth_is_yes_wholly_in_view_unclear_in_part_and_no_otherwise():
    from handoff.tasks.capsules import block_truth

    caps = [SeenCapsule((0.0, 0.0, 0.83), (10, 10, 22, 16), 72), SeenCapsule((0.0, 0.0, 0.83), (10, 30, 22, 36), 72)]
    window = (0, 0, 40, 33)  # all of capsule 0, the first three columns of capsule 1
    assert block_truth(caps, window, leaky={0}) == "yes"
    assert block_truth(caps, window, leaky={1}) is None  # only a sliver of the leaker shows: left out of scoring
    assert block_truth(caps, window, leaky=set()) == "no"
    assert block_truth(caps, (0, 0, 40, 28), leaky={1}) == "no"  # not in the picture at all


def test_a_failed_pick_is_recorded_with_its_reason(monkeypatch):
    import handoff.tasks.capsules as task
    from handoff.capsule_skills import MoveResult

    monkeypatch.setattr(task, "pick_capsule", lambda *a, **k: MoveResult(False, "hand missed the capsule by 33 mm", 0.033))
    sim, ctx, result = run(with_leakers(1, "a"))
    failures = result.checks["move_failures"]
    assert failures and all(f["move"] == "pick" and f["reason"] == "hand missed the capsule by 33 mm" for f in failures)
    assert result.checks["leakers_in_bin"] == 0


def test_base_drift_is_read_after_a_failed_pick_too(monkeypatch):
    """Review of fdfd388: the drift sensor ran only after a drop, so shifts whose picks all failed logged 0 mm."""
    import handoff.tasks.capsules as task
    from handoff.capsule_skills import MoveResult

    def shove(sim, *a, **k):
        x, y, yaw = sim._base
        sim._base = np.array([x - 0.02, y, yaw + 0.05])  # pushed back 20 mm and turned
        return MoveResult(False, "hand missed the capsule by 40 mm", 0.04)

    monkeypatch.setattr(task, "pick_capsule", shove)
    sim, ctx, result = run(with_leakers(1, "a"))
    assert result.checks["base_drift_mm"] >= 20.0 - 1e-6
    assert result.checks["base_drift_deg"] >= np.degrees(0.05) - 1e-6


def test_a_leaker_dropped_beside_the_bin_fails_the_shift(monkeypatch):
    """Seed 1001 (Nov 5): a capsule struck the bin's wall and lay on the table; tray_clear said yes and the shift read
    success. A leaker in no bin and not on the tray now fails the shift and is named in the checks."""
    import handoff.tasks.capsules as task
    from handoff.capsule_skills import MoveResult

    def drop_on_table(sim, cell, name, arm="left"):
        sim.detach(name)
        x, y = cell.to_world(0.30, 0.55)  # on the table past the bin
        sim.set_object_pose(name, (x, y, cell.layout.table_top + cell.layout.capsule[2] / 2.0), 0.0)
        return MoveResult(True, "", 0.0)

    monkeypatch.setattr(task, "drop_in_bin", drop_on_table)
    sim, ctx, result = run(with_leakers(1, "a"))
    assert result.status is Status.FAILURE
    assert [d["object"] for d in result.checks["leakers_loose"]] == [ctx.cells.leaker_names[0]]


def test_a_leaker_dropped_twice_is_counted_once(monkeypatch):
    """Seed 1003 (Nov 6): a leaker bounced off the bin's rim back onto the tray, was picked again and binned, and the
    shift counted 4 leakers in the bin out of 3."""
    import handoff.tasks.capsules as task

    real = task.drop_in_bin
    calls = []

    def drop(sim, cell, name, arm="left"):
        calls.append(name)
        if len(calls) == 1:  # the first drop bounces back to where it was picked
            sim.detach(name)
            x, y = cell.to_world(*cell.layout.grid()[ctx_spot[0]])
            sim.set_object_pose(name, (x, y, cell.layout.paper_top + cell.layout.capsule[2] / 2.0), 0.0)
            from handoff.capsule_skills import MoveResult
            return MoveResult(True, "", 0.0)
        return real(sim, cell, name, arm)

    episode = with_leakers(1, "a")
    ctx_spot = [episode.leakers[0]]
    monkeypatch.setattr(task, "drop_in_bin", drop)
    sim, ctx, result = run(episode)
    assert len(calls) == 2 and result.checks["leakers_in_bin"] == 1


def test_a_block_of_16_is_one_whole_half_tray():
    """Nov 6 smoke run: counting columns by pitch put the aisle's width into the count and split the left half across
    two blocks (3 blocks a tray). Each half-tray is one block of 16."""
    cells = TwoCells()
    for name in ("a", "b"):
        cell = cells.cell(name)
        caps = [SeenCapsule((*cell.to_world(x, y), 0.83), (0, 0, 1, 1), 0) for x, y in cells.layout.grid()]
        groups = blocks(caps, CapsuleContext(cells=cells, episode=CapsuleEpisode(0, name, (), ()), ladder=Ladder([]), cell=name, block=16))
        assert sorted(len(g) for g in groups) == [16, 16]
        for g in groups:  # each block on one side of the aisle
            sides = {cells.layout.grid()[caps.index(c)][1] > 0 for c in g}
            assert len(sides) == 1
