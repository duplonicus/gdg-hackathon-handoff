"""The film's scripted operator (Nov 7): the plan says what each page question gets, and nothing else is answered."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("film_answerer", ROOT / "scripts" / "film_answerer.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)

OPTS = ["good", "defective"]


def test_plan_items_by_number_and_default():
    plan = fa.parse_plan("1=defective@4, *=model@2.5")
    assert plan == {"1": ("defective", 4.0), "*": ("model", 2.5)}
    assert fa.decide(plan, 1, OPTS, "") == ("defective", 4.0)
    assert fa.decide(plan, 2, OPTS, "d3-27b-vision good 0.62") == ("good", 2.5)


def test_unlisted_questions_and_none_are_left_alone():
    plan = fa.parse_plan("2=none")
    assert fa.decide(plan, 1, OPTS, "")[0] is None
    assert fa.decide(plan, 2, OPTS, "")[0] is None


def test_an_option_the_question_does_not_have_is_refused():
    with pytest.raises(ValueError):
        fa.decide(fa.parse_plan("1=grasp"), 1, OPTS, "")


def test_bad_plan_text_is_refused():
    for bad in ("defective@4", "x=good", "1="):
        with pytest.raises(ValueError):
            fa.parse_plan(bad)


def test_model_pick_reads_the_climbed_line():
    assert fa.model_pick("d3-27b-vision defective 0.60", OPTS) == "defective"
    assert fa.model_pick("d3-27b-vision: unreachable", OPTS) is None


def test_model_pick_takes_the_last_rung_before_the_page():
    assert fa.model_pick("decider-2b yes 0.55, gemma4:e4b no 0.62", ["yes", "no"]) == "no"


def test_keys_after_d3_stops_count_from_the_stop():
    plan = fa.parse_plan("*=model@4,s1=pick:defective|grasp|yes@5,s2=none,s*=pick:defective|grasp|yes@3")
    assert fa.decide(plan, 1, OPTS, "d3-27b-vision good 0.6") == ("good", 4.0)  # before the stop
    assert fa.decide(plan, 3, OPTS, "", since_stop=1) == ("defective", 5.0)
    assert fa.decide(plan, 4, ["grasp", "re_look", "ask_operator", "abort"], "", since_stop=2) == (None, 0.0)
    assert fa.decide(plan, 5, ["yes", "no"], "", since_stop=3) == ("yes", 3.0)


def test_pick_with_nothing_offered_answers_nothing():
    assert fa.decide(fa.parse_plan("*=pick:maybe"), 1, OPTS, "")[0] is None
    with pytest.raises(ValueError):
        fa.parse_plan("1=pick:")


def test_truth_answers_the_simulator_answer_when_it_is_an_option():
    plan = fa.parse_plan("*=truth@3")
    assert fa.decide(plan, 1, OPTS, "", truth="defective") == ("defective", 3.0)
    assert fa.decide(plan, 1, OPTS, "", truth=None)[0] is None  # no hint: leave it to time out
