"""The head follows the work: it faces the pick station, then the bin, and never goes back to level mid-sort."""

from pathlib import Path

import numpy as np

from handoff.decider import Ladder, Step
from handoff.logbook import Logbook
from handoff.rungs import OracleRung
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import TRAVEL_POSE, look_at
from handoff.tasks.sort import SortContext, run_sort, truth


def test_look_at_turns_left_and_down_for_a_point_on_the_left_table():
    sim = FakeSimulator()
    sim.build(make_scene(Layout()))
    base = sim.base_pose().pos
    look_at(sim, (base[0] + 0.5, base[1] + 0.5, 0.82))
    yaw, pitch = sim._head
    assert np.isclose(yaw, -np.pi / 4, atol=0.01)  # the head_yaw joint is positive to the robot's right
    assert 0.5 < pitch < 0.9  # positive pitch is down


def test_head_is_on_the_bin_at_every_bin_question(tmp_path: Path):
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 3)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    seen = []

    def answer(q):
        seen.append((q.id, sim._head))
        return truth(sim, holder["ctx"], holder["part"], q.id)

    ctx = SortContext(layout=layout, episode=episode, ladder=Ladder([Step(OracleRung(answer), 0.0)]), logbook=Logbook(tmp_path, "ep3"))
    holder["ctx"] = ctx
    import handoff.tasks.sort as sort_module

    original = sort_module.ask

    def ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return original(sim_, ctx_, qid, part)

    sort_module.ask = ask
    try:
        run_sort(sim, ctx)
    finally:
        sort_module.ask = original
    assert seen
    for qid, (yaw, pitch) in seen:
        assert pitch > 0.3, (qid, pitch)  # never level while there is work to watch
        if qid == "in_bin":
            assert yaw < -0.3, (qid, yaw)  # both bins are on the robot's left, which is negative head_yaw
