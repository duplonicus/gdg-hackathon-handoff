"""probe_wrong_tray.py: the rewritten state keeps the target and drops the claim, scoring and resume count honestly."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("probe_wrong_tray", SCRIPTS / "probe_wrong_tray.py")
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)

LOGGED = (
    "The hand released the part over the rework bin, the orange (left) tray, about 1.5 s ago and is now open. "
    "In the camera over the bins, the orange tray on the left is the rework bin and the green tray on the right is the good bin. "
    "Parts are painted blue; bare metal is grey."
)


def decision(where, intended="rework", question="in_bin", frame="f.png"):
    return {"question": question, "frame": frame, "state_text": LOGGED, "episode": "ep1", "truth": "no",
            "context": {"part": "p", "intended_bin": intended, "parts": {"p": {"in": where}}}}


def test_neutral_state_names_the_target_and_drops_the_release_claim():
    out = pw.neutral_state(LOGGED)
    assert "released the part over" not in out
    assert out.startswith("This part was meant for the rework bin, the orange (left) tray. ")
    assert "about 1.5 s ago and is now open. " in out
    assert out.endswith("Parts are painted blue; bare metal is grey.")


def test_neutral_state_refuses_unknown_wording():
    with pytest.raises(ValueError):
        pw.neutral_state("Something else entirely.")


def test_baseline_prompt_is_the_production_prompt_untouched():
    question, state = pw.prompt_for("baseline", LOGGED)
    assert question is pw.QUESTIONS["in_bin"]
    assert state == LOGGED


def test_new_criteria_cover_the_other_bin_and_production_does_not():
    question, _ = pw.prompt_for("criteria", LOGGED)
    assert "other bin" in question.criteria["no"]
    assert question.options == pw.QUESTIONS["in_bin"].options
    assert "other bin" not in pw.QUESTIONS["in_bin"].criteria["no"]  # the gap this probe exists for


@pytest.mark.parametrize(("where", "group"), [(["good"], "wrong bin"), (["rework"], "intended bin"), ([], "no bin"), (None, "no bin")])
def test_where_part_was(where, group):
    assert pw.where_part_was(decision(where)) == group


def write_log(root, name, decisions, frames=("f.png",)):
    log = root / name
    log.mkdir(parents=True)
    (log / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n", encoding="utf-8")
    for frame in frames:
        (log / frame).write_bytes(b"png")


def test_collect_puts_wrong_bin_first_and_counts_what_it_left_out(tmp_path):
    write_log(tmp_path, "a", [decision(["rework"]), decision(["good"]), decision(["good"], question="finish"), decision([], frame="gone.png")])
    items, skipped = pw.collect(tmp_path, ("a",))
    assert [item["group"] for item in items] == ["wrong bin", "intended bin"]
    assert [item["key"] for item in items] == ["a/1", "a/0"]
    assert skipped == {"a: no frame": 1}


def test_collect_refuses_the_spent_log(tmp_path):
    with pytest.raises(SystemExit):
        pw.collect(tmp_path, ("test04",))


@pytest.mark.parametrize(("probs", "expected"), [
    ({"yes": 0.99, "no": 0.01}, "said yes"),
    ({"yes": 0.80, "no": 0.20}, "said yes"),
    ({"yes": 0.79, "no": 0.21}, "handed off"),
    ({"yes": 0.1, "no": 0.9}, "said no"),
])
def test_outcome_at_the_frozen_threshold(probs, expected):
    assert pw.outcome(probs) == expected


def test_score_splits_outcomes_counts_landings_and_missing():
    items = [
        {"key": "a/0", "group": "wrong bin", "landing": "a/ep1/p"},
        {"key": "a/1", "group": "wrong bin", "landing": "a/ep1/p"},
        {"key": "a/2", "group": "wrong bin", "landing": "a/ep2/p"},
    ]
    answers = {("a/0", "criteria"): {"probs": {"yes": 0.9, "no": 0.1}}, ("a/1", "criteria"): {"probs": {"yes": 0.1, "no": 0.9}}}
    row = pw.score(items, answers)["criteria"]["wrong bin"]
    assert row == {"decisions": 3, "answered_landings": 1, "missing": 1, "said yes": 1, "handed off": 0, "said no": 1}
    assert pw.score(items, answers)["baseline"]["wrong bin"]["missing"] == 3


def test_read_answers_retries_errors_and_keeps_the_latest(tmp_path):
    path = tmp_path / "answers.jsonl"
    rows = [
        {"key": "a/0", "variant": "criteria", "probs": {"yes": 1.0, "no": 0.0}, "error": None},
        {"key": "a/1", "variant": "criteria", "probs": {}, "error": "timed out"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert set(pw.read_answers(path)) == {("a/0", "criteria")}


def test_finish_writes_the_marker_only_when_complete(tmp_path):
    items = [{"key": "a/0", "group": "wrong bin", "landing": "x"}]
    marker = tmp_path / "results.json"
    partial = {("a/0", "baseline"): {"probs": {"yes": 1.0, "no": 0.0}}}
    assert pw.finish(items, partial, marker) == 1
    assert not marker.exists()
    full = {("a/0", v): {"probs": {"yes": 1.0, "no": 0.0}} for v in pw.VARIANTS}
    assert pw.finish(items, full, marker) == 0
    assert json.loads(marker.read_text())["table"]["baseline"]["wrong bin"]["said yes"] == 1


def test_tray_counts_leave_out_the_part_being_asked_about():
    d = decision(["good"])
    d["context"]["parts"].update({"q": {"in": ["rework"]}, "r": {"in": ["rework"]}, "s": {"in": []}})
    assert pw.tray_counts(d) == {"rework": 2}


def test_count_state_names_both_trays_and_keeps_the_neutral_wording():
    out = pw.count_state(LOGGED, {"rework": 1})
    assert out.startswith("This part was meant for the rework bin")
    assert out.endswith("Before this release the green (right) tray held 0 parts and the orange (left) tray held 1 part.")


def test_collect_records_occupancy_and_an_earlier_parts_frame(tmp_path):
    first = decision(["rework"])
    first["context"]["part"] = "q"
    first["context"]["parts"] = {"q": {"in": ["rework"]}}
    second = decision(["good"])
    second["context"]["parts"] = {"p": {"in": ["good"]}, "q": {"in": ["rework"]}}
    write_log(tmp_path, "a", [first, second, second])
    items = {item["key"]: item for item in pw.collect(tmp_path, ("a",))[0]}
    assert items["a/0"]["before"] is None and items["a/0"]["target_occupied"] is False
    assert items["a/1"]["target_occupied"] is True and items["a/1"]["counts"] == {"rework": 1}
    assert items["a/1"]["before"].endswith("f.png")
    assert items["a/2"]["before"] == items["a/1"]["before"]  # asking the same part again still compares with the other part


def test_before_after_applies_only_with_an_earlier_frame_and_is_not_counted_missing():
    items = [{"key": "a/0", "group": "wrong bin", "landing": "x", "before": None}]
    answers = {("a/0", v): {"probs": {"yes": 1.0, "no": 0.0}} for v in ("baseline", "criteria", "criteria_neutral", "counts")}
    table = pw.score(items, answers)
    assert table["before_after"]["wrong bin"] == {"decisions": 0, "answered_landings": 0, "missing": 0, "said yes": 0, "handed off": 0, "said no": 0}
    assert table["counts"]["wrong bin"]["missing"] == 0


def test_two_frames_are_sent_in_order():
    seen = {}

    def post(url, body, timeout):
        seen.update(url=url, body=body)
        return {"answers": {"in_bin": {"probabilities": {"yes": 0.1, "no": 0.9}}}}

    question, _ = pw.prompt_for("criteria", LOGGED)
    answer = pw.ask_two_frames("http://x", question, "state", ["AAA", "BBB"], post)
    assert answer.top == "no"
    assert seen["body"]["images"] == ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]


def test_occupancy_splits_by_the_target_tray():
    occupied = decision(["good"])
    occupied["context"]["parts"]["q"] = {"in": ["rework"]}
    empty = decision(["good"])
    right = decision(["rework"])
    one = {"decisions": 1, "landings": 1}
    assert pw.occupancy([occupied, empty, right], {0: 0.99, 1: 0.1, 2: 0.99}) == {("target occupied", "said yes"): one, ("target empty", "said no"): one}


def test_occupancy_counts_a_landing_asked_twice_once():
    occupied = decision(["good"])
    occupied["context"]["parts"]["q"] = {"in": ["rework"]}
    assert pw.occupancy([occupied, occupied], {0: 0.99, 1: 0.99}) == {("target occupied", "said yes"): {"decisions": 2, "landings": 1}}


@pytest.mark.parametrize(("variant", "expected"), [
    ("before_after", ["b", "n"]), ("ba_swapped", ["n", "b"]), ("ba_same_twice", ["n", "n"]), ("ba_one_frame", ["n"]),
])
def test_each_frame_variant_sends_its_own_frames(variant, expected):
    assert pw.frames_for(variant, "b", "n") == expected


def test_frame_controls_differ_from_before_after_only_in_the_frames():
    item = {"state": LOGGED, "before": "x"}
    reference = pw.item_prompt("before_after", item)
    for variant in pw.FRAME_VARIANTS:
        assert pw.item_prompt(variant, item) == reference
        assert pw.applies(variant, item)
        assert not pw.applies(variant, {"state": LOGGED, "before": None})


def test_baseline_drift_compares_only_this_logs_baseline():
    answers = {("a/1", "baseline"): {"probs": {"yes": 0.5}}, ("a/2", "baseline"): {"probs": {"yes": 0.9}}, ("a/1", "criteria"): {"probs": {"yes": 0.0}}, ("b/1", "baseline"): {"probs": {"yes": 0.0}}}
    compared, largest = pw.baseline_drift(answers, "a", {1: 0.5, 2: 0.7, 3: 0.1})
    assert compared == 2
    assert largest == pytest.approx(0.2)


def test_finish_refuses_a_baseline_that_drifted(tmp_path):
    log = tmp_path / "a"
    log.mkdir()
    (log / "arms.jsonl").write_text(json.dumps({"arm": "d3-27b-vision", "question": "in_bin", "index_global": 0, "probs": {"yes": 0.99}, "error": None}) + "\n", encoding="utf-8")
    items = [{"key": "a/0", "log": "a", "group": "wrong bin", "landing": "x", "before": "f"}]
    answers = {("a/0", v): {"probs": {"yes": 0.1, "no": 0.9}} for v in pw.VARIANTS}
    marker = tmp_path / "results.json"
    assert pw.finish(items, answers, marker, tmp_path) == 1
    assert not marker.exists()
