"""Return-home is a switch, and the sort stops with a reason when it cannot find a bin."""

from pathlib import Path

import numpy as np

import handoff.skills as skills
import handoff.tasks.sort as sort_module
from handoff.bt import Status
from handoff.decider import Ladder, Step
from handoff.logbook import Logbook
from handoff.rungs import OracleRung
from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import TRAVEL_POSE
from handoff.tasks.sort import SortContext, run_sort, truth


def run(tmp_path: Path, monkeypatch, seed: int = 0, sim: FakeSimulator | None = None, **options):
    layout = Layout(bin_jitter_m=0.02)
    sim = sim or FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    ladder = Ladder([Step(OracleRung(lambda q: truth(sim, holder["ctx"], holder["part"], q.id)), 0.0)])
    ctx = SortContext(layout=layout, episode=episode, ladder=ladder, logbook=Logbook(tmp_path, f"ep{seed}"), **options)
    holder["ctx"] = ctx
    original = sort_module.ask

    def ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return original(sim_, ctx_, qid, part)

    monkeypatch.setattr(sort_module, "ask", ask)
    used = []
    for name in ("go_home", "retract"):
        real = skills.SKILLS[name]
        monkeypatch.setitem(skills.SKILLS, name, lambda s, c, _real=real, _name=name: (used.append(_name), _real(s, c))[1])
    return run_sort(sim, ctx), used, ctx


def test_return_home_is_the_default_and_the_switch_replaces_it_with_retract(tmp_path, monkeypatch):
    with_home, used_home, _ = run(tmp_path / "a", monkeypatch)
    assert with_home.status is Status.SUCCESS, with_home.trace.text()
    assert used_home.count("go_home") == 3 and "retract" not in used_home


def test_no_return_home_still_sorts_every_part_and_takes_less_time(tmp_path, monkeypatch):
    with_home, _, _ = run(tmp_path / "a", monkeypatch)
    without, used, _ = run(tmp_path / "b", monkeypatch, return_home=False)
    assert without.status is Status.SUCCESS, without.trace.text()
    assert used.count("retract") == 3 and "go_home" not in used
    assert without.checks["parts_in_correct_bin"] == 3
    assert without.sim_time < with_home.sim_time


def test_sort_halts_with_a_reason_when_a_bin_is_not_in_view(tmp_path, monkeypatch):
    sim = FakeSimulator()
    real_cloud = sim.capture_cloud

    def only_the_good_bin(camera):
        rgb, xyz = real_cloud(camera)
        green = rgb[..., 1] > rgb[..., 0]  # drop every orange (rework) point: that tray is hidden
        return rgb[green][None], xyz[green][None]

    monkeypatch.setattr(sim, "capture_cloud", only_the_good_bin)
    result, _, ctx = run(tmp_path, monkeypatch, sim=sim)
    assert result.status is Status.FAILURE
    assert "could not find the rework bin" in result.trace.text()
    assert ctx.bins_seen == {}  # nothing is aimed at from a partial sighting
    assert result.checks["parts_in_any_bin"] == 0  # and nothing was carried
    assert np.isfinite(result.sim_time)
