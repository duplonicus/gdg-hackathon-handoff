"""The hand crosses the table at one clearance height, comes straight down, and learns from bumping into things.

dup, Oct 21: the arm bumped its hand on the trays carrying parts; "it needs to move up more before it moves over, or
eventually learn to correct itself". Both are checked here on the fake, which records every move target.
"""

import numpy as np
import pytest

import handoff.tasks.sort as sort_module
from handoff.bt import Status
from handoff.decider import Ladder, Step
from handoff.rungs import OracleRung
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import (
    BUMP_EXTRA_CAP,
    BUMP_EXTRA_STEP,
    BUMP_RETRIES,
    BUMP_RISE,
    LATERAL_MIN,
    SKILLS,
    TRAVEL_POSE,
    TaskContext,
    _hand_over_tray,
    clear_spot,
    clearance_z,
    fixed_boxes,
)
from handoff.tasks.sort import SortContext, run_sort, truth


def lifted(seed: int = 0, **ctx_kw) -> tuple[FakeSimulator, TaskContext]:
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    events: list[str] = []
    ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts, on_event=events.append, **ctx_kw)
    ctx.data["events"] = events
    for name in ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift"):
        assert SKILLS[name](sim, ctx).ok, name
    return sim, ctx


def lateral(move) -> bool:
    return float(np.linalg.norm(move.target[:2] - move.start[:2])) > LATERAL_MIN


def low_lateral_moves(moves, floor: float) -> list:
    """Moves that travel sideways with either end below ``floor``."""
    return [m for m in moves if lateral(m) and min(m.start[2], m.target[2]) < floor - 1e-9]


def test_clearance_is_the_wall_top_height_plus_the_held_part_plus_the_margin():
    sim, ctx = lifted()
    wall_top = ctx.layout.bin_region("good").wall_top_z
    assert abs(clearance_z(sim, ctx) - (wall_top + ctx.layout.part_size[2] / 2.0 + ctx.travel_margin)) < 1e-9  # 1.03 now
    sim.set_hand(ctx.arm, 0.0)
    sim.set_object_pose(ctx.target_part, (0.9, -0.3, ctx.layout.part_center_z), 0.0)  # empty hand, part far away
    assert abs(clearance_z(sim, ctx) - (wall_top + ctx.travel_margin)) < 1e-9


def test_move_to_bin_goes_up_first_and_never_down_on_its_first_move():
    sim, ctx = lifted()
    here = sim.wrist_pose(ctx.arm).pos.copy()
    clear = clearance_z(sim, ctx)
    assert here[2] < clear  # after the lift the wrist is at about 1.01, under the 1.03 crossing
    n = len(sim.moves)
    result = SKILLS["move_to_bin"](sim, ctx)
    assert result.ok, result.reason
    first = sim.moves[n]
    assert first.target[2] >= first.start[2]  # the old step 1 aimed at wall top + hover, 2 cm BELOW the lifted wrist
    assert np.allclose(first.target[:2], here[:2])  # straight up where it is
    assert not low_lateral_moves(sim.moves[n:], clear)


def test_move_to_bin_crosses_at_the_wrist_height_when_that_is_already_above_the_clearance():
    sim, ctx = lifted(lift_height=0.25)
    here_z = float(sim.wrist_pose(ctx.arm).pos[2])
    assert here_z > clearance_z(sim, ctx)
    n = len(sim.moves)
    assert SKILLS["move_to_bin"](sim, ctx).ok
    moves = sim.moves[n:]
    assert all(m.target[2] >= here_z - 1e-9 for m in moves if lateral(m))  # the go-up step never brings it down
    assert moves[0].target[2] >= moves[0].start[2]


def test_move_to_bin_crosses_then_comes_straight_down_to_the_hover():
    sim, ctx = lifted()
    n = len(sim.moves)
    assert SKILLS["move_to_bin"](sim, ctx).ok
    region = ctx.layout.bin_region(ctx.target_bin)
    last = sim.moves[-1]
    assert not lateral(last) and last.target[2] < last.start[2]  # the last move is the descent
    assert abs(last.target[2] - (region.wall_top_z + ctx.hover_above_bin)) < 1e-9
    assert all(set(fixed_boxes(ctx)) <= set(m.stop_on) for m in sim.moves[n:] if lateral(m))  # crossings are guarded


@pytest.mark.parametrize("skill_name", ["go_home", "retract"])
def test_going_home_rises_and_crosses_clear_of_both_trays_before_the_joint_move(skill_name):
    sim, ctx = lifted()
    for name in ("move_to_bin", "lower", "release"):
        SKILLS[name](sim, ctx)
    n = len(sim.moves)
    assert SKILLS[skill_name](sim, ctx).ok
    moves = sim.moves[n:]
    clear = clearance_z(sim, ctx)
    assert moves[0].target[2] >= moves[0].start[2]
    assert any(lateral(m) for m in moves)  # the old skills only lifted 8 cm over the bin, then snapped the joints
    assert not low_lateral_moves(moves, clear)
    wrist = sim.wrist_pose(ctx.arm).pos
    assert not _hand_over_tray(ctx, float(wrist[0]), float(wrist[1]))
    assert wrist[2] >= clear - 1e-9


def test_the_clear_spot_slides_toward_the_midline_when_a_tray_sits_over_home():
    layout = Layout()
    ctx = TaskContext(layout=layout)
    assert clear_spot(ctx) == (0.32, 0.22)  # the default trays leave the home spot clear
    ctx = TaskContext(layout=layout, bins_seen={"good": (0.42, 0.33), "rework": (0.44, 0.57)})
    x, y = clear_spot(ctx)
    assert y < 0.22 and not _hand_over_tray(ctx, x, y)


def test_approach_comes_straight_down_beside_the_part_then_slides_in():
    layout = Layout(robot_start=(0.0, 0.0, 0.0))  # already parked at the table: the arms move with the base
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 0)
    place_episode(sim, layout, episode)
    ctx = TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts)
    assert SKILLS["reach"](sim, ctx).ok
    n = len(sim.moves)
    assert SKILLS["approach"](sim, ctx).ok
    drop, slide = sim.moves[n:]  # the old approach was one diagonal: 10 cm down and 3 cm across at once
    assert np.allclose(drop.target[:2], drop.start[:2]) and drop.target[2] < drop.start[2]
    assert abs(slide.target[2] - slide.start[2]) < 1e-9 and lateral(slide)


def test_a_bump_halts_the_crossing_backs_up_and_retries_higher():
    sim, ctx = lifted()
    sim.inject_bump("good_wall_ym")
    n = len(sim.moves)
    result = SKILLS["move_to_bin"](sim, ctx)
    assert result.ok, result.reason  # it still gets there
    moves = sim.moves[n:]
    hit = next(i for i, m in enumerate(moves) if m.bump)
    crossing = moves[hit].target[2]
    rise, retry = moves[hit + 1], moves[hit + 2]
    assert not lateral(rise) and rise.target[2] >= moves[hit].start[2] + BUMP_RISE - 1e-9  # backs straight up first
    assert lateral(retry) and abs(retry.target[2] - (crossing + BUMP_RISE)) < 1e-9
    (bump,) = ctx.bumps
    # The box is the robot's estimate (nearest the hand in the drawing), not the injected truth: the fake's touch is
    # scripted by name, so where its hand is halfway decides what the robot names.
    assert bump["box"] in fixed_boxes(ctx) and ctx.data["events"].count(f"bump_{bump['box']}") == 1
    assert set(bump) == {"box", "skill", "wrist_z", "retry_z", "clearance_extra"}
    assert bump["skill"] == "move_to_bin"
    assert abs(bump["retry_z"] - (crossing + BUMP_RISE)) < 1e-4
    assert abs(bump["clearance_extra"] - BUMP_EXTRA_STEP) < 1e-9
    assert abs(ctx.clearance_extra - BUMP_EXTRA_STEP) < 1e-9


def test_a_bump_on_the_way_down_hovers_higher_instead_of_pushing_on():
    sim, ctx = lifted()
    original = sim.move_wrist
    guarded = []

    def move(arm, pos, quat, **kw):
        if kw.get("stop_on"):
            guarded.append(pos)
            if len(guarded) == 2:  # the crossing, then the descent: the descent touches the near wall
                sim.inject_bump("good_wall_xm")
        return original(arm, pos, quat, **kw)

    sim.move_wrist = move
    result = SKILLS["move_to_bin"](sim, ctx)
    assert result.ok, result.reason
    region = ctx.layout.bin_region(ctx.target_bin)
    assert abs(sim.wrist_pose(ctx.arm).pos[2] - (region.wall_top_z + ctx.hover_above_bin + BUMP_RISE)) < 1e-9
    assert [b["box"] for b in ctx.bumps] == ["good_wall_xm"]
    assert abs(ctx.bumps[0]["retry_z"] - (region.wall_top_z + ctx.hover_above_bin + BUMP_RISE)) < 1e-4  # the hover it retried at


def test_a_crossing_bump_shows_the_raised_crossing_height_in_the_result():
    sim, ctx = lifted()
    planned = clearance_z(sim, ctx)
    sim.inject_bump("good_wall_ym")
    result = SKILLS["move_to_bin"](sim, ctx)
    assert result.ok, result.reason
    assert abs(result.data["crossing_z"] - (planned + BUMP_RISE)) < 1e-9  # what it actually crossed at, not the plan


def test_lower_halts_on_a_bump_and_lets_go_there_instead_of_pushing_on():
    sim, ctx = lifted()
    assert SKILLS["move_to_bin"](sim, ctx).ok
    hover = sim.wrist_pose(ctx.arm).pos[2]
    n = len(sim.moves)
    sim.inject_bump("good_wall_xm")
    result = SKILLS["lower"](sim, ctx)
    assert result.ok, result.reason
    assert set(fixed_boxes(ctx)) <= set(sim.moves[n].stop_on)  # the descent is guarded
    assert len(sim.moves) == n + 1  # no second push down after the stop
    stopped = sim.wrist_pose(ctx.arm).pos[2]
    assert hover > stopped > ctx.layout.bin_region(ctx.target_bin).wall_top_z + ctx.release_above_wall
    assert ctx.bumps[-1]["skill"] == "lower" and ctx.bumps[-1]["retry_z"] is None
    assert result.data["bumped"] == "good_wall_xm"
    assert "bump_good_wall_xm" in ctx.data["events"]


def test_the_learned_extra_is_capped():
    ctx = TaskContext()
    ctx.bumps.extend({"box": "table"} for _ in range(20))
    assert ctx.clearance_extra == BUMP_EXTRA_CAP


def test_a_move_that_keeps_bumping_gives_up_and_reports_the_miss():
    sim, ctx = lifted()
    sim.inject_bump("good_wall_ym", times=BUMP_RETRIES + 1)
    result = SKILLS["move_to_bin"](sim, ctx)
    assert not result.ok and "missed" in result.reason
    assert len(ctx.bumps) == BUMP_RETRIES + 1


def test_a_bump_raises_the_crossing_for_the_parts_after_it(tmp_path, monkeypatch):
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 0)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    ladder = Ladder([Step(OracleRung(lambda q: truth(sim, holder["ctx"], holder["part"], q.id)), 0.0)])
    events: list[str] = []
    ctx = SortContext(layout=layout, episode=episode, ladder=ladder, on_event=events.append)
    holder["ctx"] = ctx
    original = sort_module.ask

    def ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return original(sim_, ctx_, qid, part)

    monkeypatch.setattr(sort_module, "ask", ask)
    real = SKILLS["move_to_bin"]
    crossings = []

    def move_to_bin(s, c):
        if not crossings:
            s.inject_bump("good_wall_ym")  # the first part's crossing touches a wall
        r = real(s, c)
        crossings.append(r.data["crossing_z"])
        return r

    monkeypatch.setitem(SKILLS, "move_to_bin", move_to_bin)
    result = run_sort(sim, ctx)
    assert result.status is Status.SUCCESS, result.trace.text()
    # The robot names the wall nearest its own estimate of the hand, which turns with the hand's heading (the fake's
    # hold point turned with it from Nov 4): with this part's yaw that is the next wall over, still the good tray's.
    assert len(ctx.bumps) == 1 and any(e.startswith("bump_good_wall_") for e in events)
    assert len(crossings) == 3
    planned_first = crossings[0] - BUMP_RISE  # the first part crossed after its retry, BUMP_RISE over its plan
    assert abs(crossings[1] - planned_first - BUMP_EXTRA_STEP) < 1e-9  # the next part's plan is 3 cm higher
    assert abs(crossings[2] - crossings[1]) < 1e-9  # and it stays there: no new bump, no new raise
