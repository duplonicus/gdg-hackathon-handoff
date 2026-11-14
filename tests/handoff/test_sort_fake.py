"""The sort tree on the fake simulator with the oracle answering (no models, no GPU)."""

from pathlib import Path

from handoff.decider import Ladder, Step
from handoff.rungs import OracleRung
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import TRAVEL_POSE
from handoff.tasks.sort import SortContext, run_sort, truth
from handoff.bt import Status
from handoff.logbook import Logbook, read_decisions
import handoff.tasks.sort as sort_module


def make(seed: int, tmp_path: Path, fault=None):
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    oracle = OracleRung(lambda q: truth(sim, holder["ctx"], holder["part"], q.id))
    ctx = SortContext(layout=layout, episode=episode, ladder=Ladder([Step(oracle, 0.0)]), logbook=Logbook(tmp_path, f"ep{seed}"), fault=fault)
    holder["ctx"] = ctx
    real_ask = sort_module.ask

    def tracking_ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return real_ask(sim_, ctx_, qid, part)

    sort_module.ask = tracking_ask
    return sim, ctx, real_ask


def test_sort_with_the_oracle_places_every_part_in_its_right_bin_and_dances(tmp_path):
    sim, ctx, real_ask = make(0, tmp_path)
    ctx.dance = True
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert result.status is Status.SUCCESS, result.trace.text()
    assert result.checks["parts_in_correct_bin"] == result.checks["n_parts"] == 3
    assert result.checks["danced"]
    kinds = [d.question_id for d in result.decisions]
    assert kinds.count("finish") == 3 and kinds.count("safe_to_grasp") == 3 and kinds.count("in_bin") == 3 and kinds.count("sort_complete") == 1
    assert all(d.truth == d.answer for d in result.decisions)
    rows = read_decisions(tmp_path / "decisions.jsonl")
    assert len(rows) == len(result.decisions) and rows[0]["frame"] and (tmp_path / rows[0]["frame"]).exists()


def test_forced_drop_is_detected_and_recovered(tmp_path):
    sim, ctx, real_ask = make(1, tmp_path, fault="drop_on_carry")
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert ctx.fault_fired
    in_bin_answers = [d.answer for d in result.decisions if d.question_id == "in_bin"]
    assert "no" in in_bin_answers, in_bin_answers
    assert "recover: success" in result.trace.text() or "re-pick and place: success" in result.trace.text()
    assert result.status is Status.SUCCESS, result.trace.text()
    assert result.checks["parts_in_correct_bin"] == 3


def test_crowded_station_makes_the_rule_say_abort_and_the_part_is_rejected(tmp_path):
    sim, ctx, real_ask = make(2, tmp_path, fault="crowded")
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert ctx.fault_fired
    first = ctx.episode.parts[0]
    safe = [d for d in result.decisions if d.question_id == "safe_to_grasp" and d.context["part"] == first]
    assert safe and safe[0].answered_by == "rule" and safe[0].answer == "abort" and safe[0].truth == "abort"
    assert ctx.intended_bin[first] == "reject"
    assert result.status is Status.FAILURE  # a rejected part means the sort cannot complete, and no dance
    assert not result.checks["danced"]
    assert result.checks["parts_in_correct_bin"] == 2


def test_a_part_dropped_again_after_one_re_pick_goes_to_the_operator(tmp_path, monkeypatch):
    """Oct 21 demo: part_12 fell out of the hand on every carry and the robot re-picked it four times. One robot
    re-pick per part; after that the operator places it."""
    sim, ctx, real_ask = make(0, tmp_path)
    first = ctx.episode.parts[0]
    events = []
    ctx.on_event = events.append
    real_wait = sort_module.wait_until_resting

    def drops(sim_, tctx, **kw):  # the part ends on the table beside the station after every robot carry
        if tctx.target_part == first:
            x, y, z = ctx.layout.slot_pos(0)
            sim_.set_object_pose(first, (x + 0.06, y, z), 0.0)
        return real_wait(sim_, tctx, **kw)

    monkeypatch.setattr(sort_module, "wait_until_resting", drops)
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert events.count("recover_lift") == 1 and events.count("operator_place") == 1, events
    assert ctx.robot_repicks == {first: 1}
    assert result.checks["operator_moves"] == 1 and result.checks["parts_in_correct_bin"] == 3, result.trace.text()
    assert result.status is Status.SUCCESS


def test_no_dance_unless_asked_and_a_dance_backs_away_from_the_table_first(tmp_path):
    """Oct 21, dup: no dancing until he is happy with everything, and it danced into the desk."""
    sim, ctx, real_ask = make(0, tmp_path)
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert result.status is Status.SUCCESS and not result.checks["danced"]
    assert "macarena" not in result.trace.text()

    sim, ctx, real_ask = make(0, tmp_path / "dance")
    ctx.dance = True
    first_beat = {}
    real_set = sim.set_joints

    def set_joints(targets):
        if "head_pitch" in targets and "x" not in first_beat:  # the dance's first beat: arms straight out
            first_beat["x"] = float(sim.base_pose().pos[0])
        return real_set(targets)

    sim.set_joints = set_joints
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    from handoff.skills import DANCE_CLEAR_M

    assert result.checks["danced"]
    assert first_beat["x"] <= ctx.layout.table_x[0] - DANCE_CLEAR_M + 0.02


def test_a_weld_is_released_when_the_lift_after_it_fails(tmp_path, monkeypatch):
    """The Oct 21 review: a failed lift after a touch-grip weld left the part riding the wrist into the reject lane."""
    from handoff.skills import SKILLS, SkillResult, grip_settings

    sim, ctx, real_ask = make(0, tmp_path)
    for key, value in grip_settings("touch").items():
        setattr(ctx, key, value)
    first = ctx.episode.parts[0]
    real_lift = SKILLS["lift"]
    failed = []

    def lift(sim_, tctx):
        if tctx.target_part == first and not failed:
            failed.append(first)
            return SkillResult(False, "part did not come up with the hand")
        return real_lift(sim_, tctx)

    monkeypatch.setitem(SKILLS, "lift", lift)
    try:
        run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    attach = sim.calls.index(f"attach:{first}")
    assert f"detach:{first}" in sim.calls[attach:], [c for c in sim.calls if first in c]
    assert first not in sim._welded
