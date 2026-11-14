"""The ladder, the rules, the executable check and the tree (no models, no GPU)."""

import math

from handoff.bt import Action, Condition, ForEach, Retry, Selector, Sequence, Status, Trace
from handoff.decider import QUESTIONS, Ladder, Question, RungAnswer, Step
from handoff.rungs import OracleRung, probs_from_top_logprobs


class FixedRung:
    def __init__(self, name, probs, error=None):
        self.name, self.probs, self.error = name, probs, error
        self.calls = 0

    def answer(self, question, state_text, image_b64):
        self.calls += 1
        return RungAnswer(self.name, dict(self.probs), 1.0, self.error)


FINISH = QUESTIONS["finish"]
SAFE = QUESTIONS["safe_to_grasp"]


def all_ok(question: Question) -> dict[str, bool]:
    return {o: True for o in question.options}


def test_confident_first_rung_is_accepted_and_higher_rungs_are_not_called():
    low = FixedRung("small", {"good": 0.95, "defective": 0.05})
    high = FixedRung("big", {"good": 0.1, "defective": 0.9})
    ladder = Ladder([Step(low, 0.9), Step(high, 0.6)])
    d = ladder.decide(FINISH, "state", None, all_ok)
    assert d.answer == "good" and d.answered_by == "small"
    assert high.calls == 0 and len(d.rungs) == 1


def test_unconfident_answer_escalates_and_every_rung_is_logged():
    low = FixedRung("small", {"good": 0.5, "defective": 0.5})
    high = FixedRung("big", {"good": 0.1, "defective": 0.9})
    d = Ladder([Step(low, 0.9), Step(high, 0.6)]).decide(FINISH, "state", None, all_ok)
    assert d.answer == "defective" and d.answered_by == "big"
    assert [r.rung for r in d.rungs] == ["small", "big"]


def test_nobody_confident_falls_to_the_safe_option():
    low = FixedRung("small", {"grasp": 0.6, "re_look": 0.2, "ask_operator": 0.1, "abort": 0.1})
    d = Ladder([Step(low, 0.9)]).decide(SAFE, "state", None, all_ok)
    assert d.answer == "ask_operator" and d.answered_by == "safe_option"


def test_non_executable_top_option_is_skipped():
    rung = FixedRung("small", {"grasp": 0.7, "re_look": 0.25, "ask_operator": 0.03, "abort": 0.02})

    def cannot_grasp(question):
        return {o: (o != "grasp") for o in question.options}

    d = Ladder([Step(rung, 0.2)]).decide(SAFE, "state", None, cannot_grasp)
    assert d.answer == "re_look"
    assert d.executable["grasp"] is False


def test_failed_rung_escalates_instead_of_crashing():
    broken = FixedRung("small", {"good": 1.0, "defective": 0.0}, error="ConnectionError: down")
    big = FixedRung("big", {"good": 0.0, "defective": 1.0})
    d = Ladder([Step(broken, 0.5), Step(big, 0.5)]).decide(FINISH, "state", None, all_ok)
    assert d.answer == "defective" and d.rungs[0].error is not None


def test_rule_overrides_every_rung_for_safety_questions():
    rung = FixedRung("small", {"grasp": 0.99, "re_look": 0.01, "ask_operator": 0.0, "abort": 0.0})
    ladder = Ladder([Step(rung, 0.5)], rules=lambda q: "abort" if q.kind == "safety" else None)
    d = ladder.decide(SAFE, "state", None, all_ok)
    assert d.answer == "abort" and d.answered_by == "rule" and rung.calls == 0


def test_oracle_rung_answers_the_truth():
    rung = OracleRung(lambda q: "defective")
    a = rung.answer(FINISH, "", None)
    assert a.top == "defective" and a.confidence == 1.0


def test_probs_from_first_token_logprobs_merge_prefix_tokens():
    top = [
        {"token": "def", "logprob": math.log(0.6)},
        {"token": "defect", "logprob": math.log(0.1)},
        {"token": "un", "logprob": math.log(0.2)},  # starts no option now: dropped
        {"token": "Good", "logprob": math.log(0.05)},
        {"token": "the", "logprob": math.log(0.05)},
    ]
    probs = probs_from_top_logprobs(top, FINISH.options)
    assert abs(probs["defective"] - 0.7 / 0.75) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_foreach_gives_every_item_its_turn_and_records_failures():
    bb = {"parts": ["a", "b", "c"]}
    node = ForEach("parts", "parts", "part", Action("odd", lambda bb: bb["part"] != "b"))
    assert node.tick(bb, Trace()) is Status.SUCCESS
    assert bb["parts_failed"] == ["b"]
    strict = ForEach("parts", "parts", "part", Action("odd", lambda bb: bb["part"] != "b"), fail_fast=True)
    assert strict.tick(bb, Trace()) is Status.FAILURE


def test_tree_sequence_selector_retry_and_foreach():
    bb = {"parts": ["a", "b"], "fails_left": 1}

    def flaky(bb):
        if bb["fails_left"] > 0:
            bb["fails_left"] -= 1
            return False
        return True

    done = []
    tree = Sequence(
        "root",
        ForEach("parts", "parts", "part", Sequence("per_part", Action("note", lambda bb: done.append(bb["part"]) or True))),
        Selector("fallback", Condition("never", lambda bb: False), Retry("retry", Action("flaky", flaky), 3)),
    )
    trace = Trace()
    assert tree.tick(bb, trace) is Status.SUCCESS
    assert done == ["a", "b"]
    assert "flaky: failure" in trace.text() and "retry: success" in trace.text()


def test_step_threshold_can_be_per_question():
    rung = FixedRung("small", {"good": 0.88, "defective": 0.12})
    strict = Ladder([Step(rung, {"finish": 0.9})]).decide(FINISH, "s", None, all_ok)
    loose = Ladder([Step(rung, {"finish": 0.8})]).decide(FINISH, "s", None, all_ok)
    assert strict.answered_by == "safe_option" or strict.answered_by != "small"
    assert loose.answered_by == "small"


def test_every_question_has_a_conservative_safe_option() -> None:
    from handoff.decider import QUESTIONS

    for question in QUESTIONS.values():
        assert question.safe_option in question.options, question.id
    # An unanswered finish question must not send the part to the good bin.
    assert QUESTIONS["finish"].safe_option != "good"


def test_no_question_has_two_options_with_the_same_first_word():
    """A model read by its first token's probability cannot split options that start alike ("cell_a", "cell_b")."""
    import re

    for question in QUESTIONS.values():
        firsts = [re.split(r"[_\s]", o.lower())[0] for o in question.options]
        assert len(set(firsts)) == len(firsts), question.id
        for a in question.options:
            for b in question.options:
                assert a == b or not b.lower().startswith(a.lower()), (question.id, a, b)
