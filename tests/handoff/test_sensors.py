"""Tray scales: the in_bin rule, the simulated load cell, the scale rung, and the sort tree using them (no GPU)."""

from pathlib import Path

import numpy as np
import pytest

import handoff.tasks.sort as sort_module
from handoff.benchmark import compose_ladder, scale_answers
from handoff.bt import Status
from handoff.decider import QUESTIONS, Ladder, RungAnswer, Step
from handoff.logbook import Logbook, read_decisions
from handoff.rungs import OracleRung
from handoff.scene import Layout, make_episode, make_scene, place_episode, tray_box_names
from handoff.sensors import (
    LoadCellSpec,
    LoadCellWatch,
    ScaleRung,
    TrayScales,
    load_cell_line,
    load_cell_rows,
    scale_check_line,
    scale_check_row,
    scale_verdict,
    tray_grams,
)
from handoff.sim.fake import FakeSimulator
from handoff.skills import TRAVEL_POSE
from handoff.tasks.sort import SortContext, run_sort, truth

PART = 51.2  # grams: 4 x 4 x 8 cm at 400 kg/m^3


# ----- the rule -----


@pytest.mark.parametrize(
    ("after", "intended", "expected"),
    [
        ({"good": 151.0, "rework": 50.0}, "good", ("yes", "intended")),
        ({"good": 100.0, "rework": 101.0}, "good", ("no", "other bin")),  # the wrong-bin landing the camera misses
        ({"good": 101.0, "rework": 50.0}, "good", ("no", "no bin")),  # dropped short: neither tray moved
        ({"good": 120.0, "rework": 50.0}, "good", (None, "unclear")),  # 20 g: a part leaning on a rim
        ({"good": 110.0, "rework": 50.0}, "good", ("no", "no bin")),  # 10 g: noise, nothing landed
        ({"good": 151.0, "rework": 101.0}, "good", (None, "unclear")),  # both trays rose
        ({"good": 200.0, "rework": 50.0}, "good", (None, "unclear")),  # two parts' worth
    ],
)
def test_scale_verdict(after, intended, expected):
    tare = {"good": 100.0, "rework": 50.0}
    assert scale_verdict(tare, after, intended, PART) == expected


def test_scale_verdict_holds_under_the_stated_noise():
    """Tare and reading each carry sigma 2.5 g: a settled part is never called wrong, and is unclear well under 1% of the time."""
    scales = TrayScales(Layout(), seed=1)
    answers = []
    for _ in range(2000):
        tare = scales.noisy({"good": 100.0, "rework": 50.0})
        after = scales.noisy({"good": 100.0 + PART, "rework": 50.0})
        answers.append(scale_verdict(tare, after, "good", PART)[0])
    assert "no" not in answers
    assert answers.count(None) <= 10


# ----- the load cell on the fake simulator -----


def built(layout=None):
    layout = layout or Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 0)
    place_episode(sim, layout, episode)
    return sim, layout, episode


def test_tray_load_weighs_a_resting_part_and_not_a_held_one():
    sim, layout, episode = built()
    part = episode.parts[0]
    region = layout.bin_region("rework")
    assert sim.tray_load(tray_box_names("rework")) == 0.0
    sim.set_object_pose(part, (region.cx, region.cy, region.floor_z + 0.2), 0.0)
    sim.settle(0.5)
    assert sim.tray_load(tray_box_names("rework")) == pytest.approx(PART / 1000 * 9.81, rel=1e-3)
    assert sim.tray_load(tray_box_names("good")) == 0.0


def test_scales_read_grams_with_the_stated_step_and_are_reproducible():
    sim, layout, episode = built()
    region = layout.bin_region("good")
    sim.set_object_pose(episode.parts[0], (region.cx, region.cy, region.floor_z + 0.1), 0.0)
    sim.settle(0.5)
    a = TrayScales(layout, seed=7).read(sim)
    b = TrayScales(layout, seed=7).read(sim)
    assert a[0] == b[0]
    assert a[1]["good"] == pytest.approx(PART, rel=1e-3) and a[1]["rework"] == 0.0
    assert all(float(v).is_integer() for v in a[0].values())  # 1 g step
    exact = TrayScales(layout, spec=LoadCellSpec(sigma_g=0.0, step_g=0.0)).read(sim)[0]
    assert exact["good"] == pytest.approx(PART, rel=1e-3)


# ----- the rung on the ladder -----


class Counting:
    name = "camera"

    def __init__(self):
        self.calls = []

    def answer(self, question, state_text, image_b64):
        self.calls.append(question.id)
        return RungAnswer(self.name, {o: 1.0 / len(question.options) for o in question.options}, 1.0)


def test_scale_rung_settles_in_bin_alone_and_unclear_climbs_on():
    reading = {"verdict": "no", "where": "other bin"}
    camera = Counting()
    ladder = Ladder([Step(ScaleRung(lambda: reading), 1.0, frozenset({"in_bin"})), Step(camera, 0.9), Step(OracleRung(lambda q: "yes"), 0.0)])
    d = ladder.decide(QUESTIONS["in_bin"], "", None, lambda q: {})
    assert (d.answer, d.answered_by) == ("no", "scale") and camera.calls == []
    reading = {"verdict": None, "where": "unclear"}
    d = ladder.decide(QUESTIONS["in_bin"], "", None, lambda q: {})
    assert d.answered_by != "scale" and camera.calls == ["in_bin"]


def test_a_rung_limited_to_in_bin_is_not_asked_other_questions():
    camera = Counting()
    ladder = Ladder([Step(camera, 0.0, frozenset({"in_bin"})), Step(OracleRung(lambda q: "good"), 0.0)])
    d = ladder.decide(QUESTIONS["finish"], "", None, lambda q: {})
    assert camera.calls == [] and d.answered_by != "camera" and [r.rung for r in d.rungs] != ["camera"]


# ----- the sort tree with scales -----


def make_sort(tmp_path: Path, seed: int = 0):
    sim, layout, _ = built()
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    oracle = OracleRung(lambda q: truth(sim, holder["ctx"], holder["part"], q.id))
    ladder = Ladder([Step(ScaleRung(lambda: holder["ctx"].scale_now), 1.0, frozenset({"in_bin"})), Step(oracle, 0.0)])
    ctx = SortContext(layout=layout, episode=episode, ladder=ladder, logbook=Logbook(tmp_path, f"ep{seed}"), scales=TrayScales(layout, seed=seed))
    holder["ctx"] = ctx
    real_ask = sort_module.ask

    def tracking_ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return real_ask(sim_, ctx_, qid, part)

    sort_module.ask = tracking_ask
    return sim, ctx, real_ask


def test_clean_sort_has_every_in_bin_answered_by_the_scales(tmp_path):
    sim, ctx, real_ask = make_sort(tmp_path)
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert result.status is Status.SUCCESS, result.trace.text()
    in_bin = [d for d in result.decisions if d.question_id == "in_bin"]
    assert len(in_bin) == 3 and all(d.answered_by == "scale" and d.answer == d.truth == "yes" for d in in_bin)
    assert all(d.context["scale"]["where"] == "intended" for d in in_bin)
    rows = read_decisions(tmp_path / "decisions.jsonl")
    logged = [r for r in rows if r["question"] == "in_bin"]
    assert logged and all(set(r["context"]["scale"]) >= {"tare_g", "reading_g", "true_g", "verdict"} for r in logged)


def test_wrong_bin_landing_is_caught_by_the_scales_and_the_operator_moves_it(tmp_path, monkeypatch):
    sim, ctx, real_ask = make_sort(tmp_path)
    first = ctx.episode.parts[0]
    real_wait = sort_module.wait_until_resting
    bounced = {"done": False}

    def bounce_into_the_other_tray(sim_, tctx, **kw):
        # the first release bounces the part into the other tray: the landing d3 called "in the bin" 37 of 37 times
        if tctx.target_part == first and not bounced["done"]:
            bounced["done"] = True
            other = next(b for b in ctx.layout.bin_names if b != tctx.target_bin)
            region = ctx.layout.bin_region(other)
            sim_.set_object_pose(first, (region.cx, region.cy, region.floor_z + 0.1), 0.0)
        return real_wait(sim_, tctx, **kw)

    monkeypatch.setattr(sort_module, "wait_until_resting", bounce_into_the_other_tray)
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert bounced["done"]
    first_in_bin = [d for d in result.decisions if d.question_id == "in_bin" and d.context["part"] == first]
    assert first_in_bin[0].answered_by == "scale" and first_in_bin[0].answer == first_in_bin[0].truth == "no"
    assert first_in_bin[0].context["scale"]["where"] == "other bin"
    assert first_in_bin[1].answered_by == "scale" and first_in_bin[1].answer == "yes"
    assert result.checks["operator_moves"] == 1
    assert result.checks["parts_in_correct_bin"] == 3 and result.status is Status.SUCCESS, result.trace.text()


# ----- the offline arm -----


def test_scale_answers_turn_logged_readings_into_an_arm_that_composes_first():
    decisions = [
        {"index_global": 0, "question": "in_bin", "truth": "no", "episode": "e", "context": {"scale": {"verdict": "no", "where": "other bin"}}},
        {"index_global": 1, "question": "in_bin", "truth": "yes", "episode": "e", "context": {"scale": {"verdict": None, "where": "unclear"}}},
        {"index_global": 2, "question": "in_bin", "truth": "yes", "episode": "e", "context": {}},
        {"index_global": 3, "question": "finish", "truth": "good", "episode": "e", "context": {}},
    ]
    scale = scale_answers(decisions)
    assert set(scale) == {0, 1}
    camera = {i: {"probs": {"yes": 0.99, "no": 0.01}, "latency_ms": 300} for i in (0, 1, 2)}
    rows = compose_ladder(decisions[:3], [("scale", scale, 1.0), ("d3", camera, 0.8)])
    assert [r["rung"] for r in rows] == ["scale", "d3", "d3"]
    assert rows[0]["answer"] == "no" == rows[0]["truth"]
    assert np.isclose(rows[1]["latency_ms"], 500)


def test_without_scales_a_wrong_bin_no_from_the_ladder_still_goes_to_the_operator(tmp_path, monkeypatch):
    """Before the scales, a part in the other tray ended the recovery ("nothing the line can re-feed")."""
    sim, ctx, real_ask = make_sort(tmp_path)
    ctx.scales = None
    first = ctx.episode.parts[0]
    real_wait = sort_module.wait_until_resting

    def bounce(sim_, tctx, **kw):
        if tctx.target_part == first and not ctx.operator_moves:
            other = next(b for b in ctx.layout.bin_names if b != tctx.target_bin)
            region = ctx.layout.bin_region(other)
            sim_.set_object_pose(first, (region.cx, region.cy, region.floor_z + 0.1), 0.0)
        return real_wait(sim_, tctx, **kw)

    monkeypatch.setattr(sort_module, "wait_until_resting", bounce)
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    first_in_bin = [d for d in result.decisions if d.question_id == "in_bin" and d.context["part"] == first]
    assert first_in_bin[0].answer == "no" and first_in_bin[0].answered_by != "scale" and "scale" not in first_in_bin[0].context
    assert result.checks["operator_moves"] == 1 and result.checks["parts_in_correct_bin"] == 3, result.trace.text()


def test_wrong_bin_fault_sets_the_first_part_in_the_other_tray_and_the_scales_catch_it(tmp_path):
    sim, ctx, real_ask = make_sort(tmp_path)
    ctx.fault = "wrong_bin"
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    first = ctx.episode.parts[0]
    assert ctx.fault_fired
    first_in_bin = [d for d in result.decisions if d.question_id == "in_bin" and d.context["part"] == first]
    assert first_in_bin[0].context["scale"]["where"] == "other bin" and first_in_bin[0].answer == first_in_bin[0].truth == "no"
    assert result.checks["operator_moves"] == 1 and result.checks["parts_in_correct_bin"] == 3, result.trace.text()


# ----- the load cells on the panel and in the log -----


def test_load_cell_rows_show_grams_and_parts_and_warn_on_a_part_that_has_not_settled():
    rows = load_cell_rows({"good": 102.9, "rework": 80.0}, PART, answering=False)
    assert rows[0] == {"label": "good tray", "value": "102.9 g", "detail": "2 parts", "state": "ok"}
    assert rows[1]["state"] == "warn" and rows[1]["detail"].startswith("1.56 parts: not a whole part")
    assert rows[2] == {"label": "in_bin", "value": "display only", "detail": "no --scales: the camera model and you answer in_bin", "state": "idle"}
    assert load_cell_rows({"good": -1.5}, PART, answering=True)[0]["detail"] == "0 parts"  # noise around an empty tray
    assert load_cell_rows({"good": 0.0}, PART, answering=True)[-1]["value"] == "scales answer"


def test_load_cell_line_puts_the_tray_that_changed_first_with_its_change():
    line = load_cell_line({"good": 0.2, "rework": 51.0}, PART, {"good": 0, "rework": 0})
    assert line == "load cells: rework 51.0 g (+1 part), good 0.2 g"
    assert load_cell_line({"good": 0.2, "rework": 0.4}, PART, {"good": 0, "rework": 2}) == "load cells: rework 0.4 g (-2 parts), good 0.2 g"
    assert load_cell_line({"good": 0.0, "rework": 0.0}, PART) == "load cells: good 0.0 g, rework 0.0 g"


def test_load_cell_watch_logs_a_count_only_once_it_has_settled():
    watch = LoadCellWatch(PART, answering=True, settle_reads=3)
    lines = [watch.update(g)[1] for g in [{"good": 0.0, "rework": 0.0}] * 3]
    assert lines == [None, None, "load cells (answering in_bin): good 0.0 g, rework 0.0 g"]
    # a part landing: the force spikes and bounces, then settles at one part
    stream = [{"good": 140.0, "rework": 0.0}, {"good": 30.0, "rework": 0.0}, {"good": 52.0, "rework": 0.0}, {"good": 49.0, "rework": 0.0}, {"good": 51.0, "rework": 0.0}, {"good": 50.0, "rework": 0.0}]
    lines = [watch.update(g)[1] for g in stream]
    assert lines == [None, None, None, None, "load cells: good 51.0 g (+1 part), rework 0.0 g", None]
    watch.extra_rows = [{"label": "last in_bin", "value": "part_01: yes", "detail": "intended (meant for good)", "state": "ok"}]
    rows, line = watch.update({"good": 51.0, "rework": 0.0})
    assert line is None and rows[-1]["label"] == "last in_bin" and rows[0]["detail"] == "1 part"


def test_a_part_leaning_on_one_tray_does_not_hide_a_part_landing_in_the_other():
    watch = LoadCellWatch(PART, settle_reads=3)
    for _ in range(3):
        watch.update({"good": 0.0, "rework": 0.0})
    # a part comes to rest leaning on the rework rim (half its weight on the tray, never settles), then one lands in good
    stream = [{"good": 0.0, "rework": 25.0}] * 2 + [{"good": 51.0, "rework": 25.0}] * 3
    lines = [watch.update(g)[1] for g in stream]
    assert lines == [None, None, None, None, "load cells: good 51.0 g (+1 part), rework 25.0 g"]
    # the leaning part slides in: rework settles at one part and is logged on its own
    lines = [watch.update({"good": 51.0, "rework": 51.0})[1] for _ in range(3)]
    assert lines == [None, None, "load cells: rework 51.0 g (+1 part), good 51.0 g"]


def test_tray_grams_reads_the_trays_without_stepping_the_sim():
    sim, layout, episode = built()
    region = layout.bin_region("rework")
    sim.set_object_pose(episode.parts[0], (region.cx, region.cy, region.floor_z + 0.1), 0.0)
    sim.settle(0.5)
    t = sim.time
    grams = tray_grams(sim, layout.bin_names)
    assert sim.time == t
    assert grams["rework"] == pytest.approx(PART, rel=1e-3) and grams["good"] == 0.0


def test_scale_check_line_says_tare_reading_gain_verdict_and_where():
    scale = {"verdict": "no", "where": "other bin", "intended": "good", "part_g": PART, "tare_g": {"good": 0.0, "rework": 1.0},
             "reading_g": {"good": 2.0, "rework": 52.0}, "true_g": {"good": 0.0, "rework": 51.2}}
    assert scale_check_line("part_01", scale) == (
        "scales: part_01 for the good tray: tare good 0 g, rework 1 g; now good 2 g, rework 52 g; "
        "gain in parts good +0.04, rework +1.00 => no (other bin); sim truth good 0.0 g, rework 51.2 g"
    )
    unclear = scale_check_line("part_01", {**scale, "verdict": None, "where": "unclear"})
    assert "=> unclear, up the ladder (unclear)" in unclear
    assert scale_check_row("part_01", scale) == {"label": "last in_bin", "value": "part_01: no", "detail": "other bin (meant for good)", "state": "warn"}


def test_benchmark_score_adds_the_scale_arm_from_a_scales_log_and_leaves_other_logs_alone():
    from handoff.benchmark import add_logged_arms

    decisions = [
        {"index_global": 0, "question": "in_bin", "truth": "yes", "episode": "e", "context": {"scale": {"verdict": "yes", "where": "target bin"}}},
        {"index_global": 1, "question": "finish", "truth": "good", "episode": "e", "context": {}},
    ]
    arms = {"d3-27b-vision": {0: {"probs": {"yes": 0.9, "no": 0.1}}}}
    assert add_logged_arms(decisions, arms) == ["scale"]
    assert set(arms) == {"d3-27b-vision", "scale"} and arms["scale"][0]["probs"] == {"yes": 1.0, "no": 0.0}
    plain = {"d3-27b-vision": {}}
    assert add_logged_arms([decisions[1]], plain) == [] and set(plain) == {"d3-27b-vision"}
