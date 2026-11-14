"""Benchmark arithmetic on synthetic decisions (no models)."""

import math

from handoff.benchmark import choose_thresholds, choose_thresholds_joint, choose_thresholds_joint_per_question, choose_thresholds_per_question, compose_ladder, expected_calibration_error, ladder_objective, rows_for_arm, rung_value, score, split


def decisions(n=20):
    return [{"index_global": i, "episode": f"ep{i % 4}", "question": "finish", "truth": ("good", "scratched", "unpainted")[i % 3]} for i in range(n)]


def arm_answers(decs, right_prob, wrong_on=(), conf=0.95):
    out = {}
    for d in decs:
        truth = d["truth"]
        pick = "unpainted" if truth == "good" else "good"
        answer = pick if d["index_global"] in wrong_on else truth
        rest = (1 - conf) / 2
        out[d["index_global"]] = {"probs": {o: (conf if o == answer else rest) for o in ("good", "scratched", "unpainted")}, "latency_ms": 100.0}
    return out


def test_score_counts_right_and_confidently_wrong():
    decs = decisions(10)
    rows = rows_for_arm(decs, arm_answers(decs, 1.0, wrong_on={1, 2}))
    rep = score("a", rows)
    assert rep.n == 10 and abs(rep.accuracy - 0.8) < 1e-9
    assert abs(rep.confidently_wrong - 0.2) < 1e-9  # conf 0.95 >= 0.9 and wrong
    assert rep.latency_ms == 100.0 and rep.human_rate == 0.0


def test_ece_is_zero_when_confidence_matches_accuracy_and_high_when_not():
    assert expected_calibration_error([0.9] * 10, [True] * 9 + [False]) < 1e-9
    assert expected_calibration_error([0.95] * 10, [False] * 10) > 0.9


def test_ladder_escalates_below_threshold_and_person_is_right_but_slow():
    decs = decisions(6)
    low = {d["index_global"]: {"probs": {"good": 0.5, "scratched": 0.3, "unpainted": 0.2}, "latency_ms": 10.0} for d in decs}
    low[0]["probs"] = {"good": 0.99, "scratched": 0.005, "unpainted": 0.005}  # confident, and right for index 0 (truth good)
    rows = compose_ladder(decs, [("low", low, 0.9)], human_latency_ms=1000.0)
    assert rows[0]["rung"] == "low" and rows[0]["latency_ms"] == 10.0 and rows[0]["answer"] == "good"
    assert all(r["human"] for r in rows[1:]) and all(r["answer"] == r["truth"] for r in rows[1:])
    assert rows[1]["latency_ms"] == 1010.0
    rep = score("ladder", rows)
    assert rep.accuracy == 1.0 and abs(rep.human_rate - 5 / 6) < 1e-9


def test_threshold_choice_rejects_a_rung_that_is_confidently_wrong():
    decs = decisions(30)
    good_rung = arm_answers(decs, 1.0, conf=0.95)  # always right at 0.95
    bad_rung = arm_answers(decs, 1.0, wrong_on=set(range(30)), conf=0.95)  # always wrong at 0.95
    t = choose_thresholds(decs, [("bad", bad_rung), ("good", good_rung)])
    assert t["bad"] == 1.0  # never trusted
    assert t["good"] <= 0.95


def test_threshold_choice_passes_up_a_mediocre_rung():
    decs = decisions(40)
    half = arm_answers(decs, 1.0, wrong_on=set(range(0, 40, 2)), conf=0.95)  # 50% right, always 0.95
    t = choose_thresholds(decs, [("half", half)])
    assert t["half"] == 1.0  # 50% right at any confidence: worse than asking the person


def test_rung_value_measures_the_upper_rung_on_escalated_cases():
    decs = decisions(12)
    low = arm_answers(decs, 1.0, wrong_on=set(range(12)), conf=0.5)  # unsure and wrong
    up = arm_answers(decs, 1.0, conf=0.9)
    v = rung_value(decs, low, 0.9, up)
    assert v["n_escalated"] == 12 and v["upper_accuracy"] == 1.0 and v["lower_accuracy_on_same"] == 0.0


def test_split_keeps_episodes_together():
    decs = decisions(20)
    held, test = split(decs, 0.25, seed=1)
    assert len(held) + len(test) == 20
    assert not ({d["episode"] for d in held} & {d["episode"] for d in test})


def test_per_question_thresholds_trust_a_rung_only_where_it_is_right():
    decs = decisions(40)
    for i, d in enumerate(decs):
        d["question"] = "finish" if i % 2 == 0 else "in_bin"
    # right on in_bin, wrong on finish, always 0.95
    wrong_on = {d["index_global"] for d in decs if d["question"] == "finish"}
    arm = arm_answers(decs, 1.0, wrong_on=wrong_on, conf=0.95)
    t = choose_thresholds_per_question(decs, [("a", arm)])
    assert t["a"]["finish"] == 1.0 and t["a"]["in_bin"] <= 0.95
    rows = compose_ladder(decs, [("a", arm, t["a"])])
    assert all(r["human"] for r in rows if r["question"] == "finish")
    assert all(not r["human"] for r in rows if r["question"] == "in_bin")


def test_collapse_finish_folds_old_three_way_logs_into_good_or_defective():
    from handoff.benchmark import collapse_finish

    decisions = [
        {"question": "finish", "truth": "unpainted", "answer": "scratched", "options": ["good", "scratched", "unpainted"]},
        {"question": "finish", "truth": "good", "answer": "good", "options": ["good", "scratched", "unpainted"]},
        {"question": "in_bin", "truth": "no", "answer": "no", "options": ["yes", "no"]},
    ]
    arms = {"m": {0: {"probs": {"good": 0.6, "scratched": 0.3, "unpainted": 0.1}}, 2: {"probs": {"yes": 0.2, "no": 0.8}}}}
    assert collapse_finish(decisions, arms) == 1
    assert decisions[0] == {"question": "finish", "truth": "defective", "answer": "defective", "options": ["good", "defective"]}
    assert decisions[1]["truth"] == "good" and decisions[1]["options"] == ["good", "defective"]
    assert decisions[2] == {"question": "in_bin", "truth": "no", "answer": "no", "options": ["yes", "no"]}
    assert arms["m"][0]["probs"] == {"good": 0.6, "defective": 0.4}
    assert arms["m"][2]["probs"] == {"yes": 0.2, "no": 0.8}
    assert collapse_finish(decisions, arms) == 0  # folding twice changes nothing


def test_rows_carry_their_episode_so_intervals_can_resample_by_episode():
    decs = decisions(8)
    rows = rows_for_arm(decs, arm_answers(decs, 1.0))
    assert [r["episode"] for r in rows] == [d["episode"] for d in decs]
    ladder = compose_ladder(decs, [("a", arm_answers(decs, 1.0, conf=0.5), 0.9)])
    assert all(r["human"] for r in ladder) and [r["episode"] for r in ladder] == [d["episode"] for d in decs]


def test_bootstrap_interval_is_by_episode_and_brackets_the_point_estimate():
    from handoff.benchmark import bootstrap_by_episode

    decs = decisions(40)  # 4 episodes of 10
    rows = rows_for_arm(decs, arm_answers(decs, 1.0, wrong_on=set(range(0, 40, 4))))  # 25% wrong, spread over episodes
    rep = score("a", rows, bootstrap=500)
    lo, hi = rep.interval["accuracy"]
    assert lo <= rep.accuracy <= hi and lo < hi
    lo, hi = rep.interval["confidently_wrong"]
    assert lo <= rep.confidently_wrong <= hi
    # a perfect arm has a degenerate interval, and so does one episode (no interval at all)
    perfect = score("p", rows_for_arm(decs, arm_answers(decs, 1.0)), bootstrap=200)
    assert perfect.interval["accuracy"] == (1.0, 1.0)
    one_episode = [dict(r, episode="only") for r in rows]
    assert bootstrap_by_episode(one_episode, 50) == {}
    # resampling by episode: the same rows with a different episode assignment give a different interval
    rows_b = [dict(r, episode=f"ep{i % 8}") for i, r in enumerate(rows)]
    assert score("b", rows_b, bootstrap=500).interval["accuracy"] != rep.interval["accuracy"]
    assert score("a", rows, bootstrap=0).interval is None


def test_judgement_rates_leave_out_the_rule_decided_questions():
    from handoff.benchmark import RULE_DECIDED

    assert RULE_DECIDED == {"safe_to_grasp"}
    decs = decisions(20)
    for i, d in enumerate(decs):
        d["question"] = "safe_to_grasp" if i % 2 == 0 else "finish"
    wrong_on = {d["index_global"] for d in decs if d["question"] == "safe_to_grasp"}
    rep = score("a", rows_for_arm(decs, arm_answers(decs, 1.0, wrong_on=wrong_on, conf=0.95)), bootstrap=0)
    assert abs(rep.accuracy - 0.5) < 1e-9 and abs(rep.confidently_wrong - 0.5) < 1e-9
    assert rep.judgement["n"] == 10 and rep.judgement["accuracy"] == 1.0 and rep.judgement["confidently_wrong"] == 0.0


def test_alone_right_counts_only_what_the_models_answered():
    decs = decisions(10)
    arm = arm_answers(decs, 1.0, conf=0.95)
    for d in decs[5:]:  # unsure on the second half: the person answers those
        arm[d["index_global"]]["probs"] = {"good": 0.4, "scratched": 0.3, "unpainted": 0.3}
    rep = score("ladder", compose_ladder(decs, [("a", arm, 0.9)]), bootstrap=0)
    assert rep.accuracy == 1.0 and abs(rep.human_rate - 0.5) < 1e-9 and abs(rep.alone_right - 0.5) < 1e-9
    plain = score("a", rows_for_arm(decs, arm), bootstrap=0)
    assert plain.alone_right == plain.accuracy  # no person in a plain arm


def test_threshold_choice_survives_a_lower_rung_that_answers_everything():
    # Executed on the real test03 log Oct 14: the Decider cleared every in_bin decision, and the Gemma rung's
    # search divided by zero on an empty remainder.
    decs = decisions(12)
    sure = arm_answers(decs, 1.0, conf=0.95)  # right and confident on all of them
    t = choose_thresholds(decs, [("sure", sure), ("next", arm_answers(decs, 1.0, conf=0.6))])
    assert t["sure"] <= 0.95 and t["next"] == 1.0
    per_q = choose_thresholds_per_question(decs, [("sure", sure), ("next", arm_answers(decs, 1.0, conf=0.6))])
    assert per_q["next"]["finish"] == 1.0


def test_joint_thresholds_pass_up_a_good_rung_when_the_next_rung_is_better():
    # arXiv 2610.21412's rule: accept a cheap answer only when it beats whatever answers instead. The greedy fit
    # compares the lower rung with the person (worth 0.5), so a 90%-right rung is trusted; the next rung is always
    # right, and only the joint fit sends the decisions there.
    decs = decisions(20)
    lower = arm_answers(decs, 0.9, wrong_on=(3, 11), conf=0.8)
    upper = arm_answers(decs, 1.0, conf=0.95)
    rungs = [("lower", lower), ("upper", upper)]
    greedy = choose_thresholds(decs, rungs)
    joint = choose_thresholds_joint(decs, rungs)
    assert greedy["lower"] <= 0.8  # greedy trusts the lower rung: 2 of 20 confidently wrong
    assert joint == {"lower": 1.0, "upper": 0.95}
    assert ladder_objective(decs, rungs, joint) == 1.0
    assert ladder_objective(decs, rungs, greedy) == (18 - 2) / 20
    rows = compose_ladder(decs, [("lower", lower, joint["lower"]), ("upper", upper, joint["upper"])])
    assert [r["rung"] for r in rows] == ["upper"] * 20 and all(r["answer"] == r["truth"] for r in rows)


def test_joint_thresholds_never_score_below_greedy_on_the_decisions_they_are_fitted_on():
    import random

    for seed in range(25):
        rng = random.Random(seed)
        decs = decisions(30)
        rungs = []
        for name in ("a", "b", "c")[: rng.choice((1, 2, 3))]:
            out = {}
            for d in decs:
                conf = rng.choice((0.4, 0.55, 0.7, 0.85, 0.95))
                answer = d["truth"] if rng.random() < conf else ("unpainted" if d["truth"] == "good" else "good")
                rest = (1 - conf) / 2
                out[d["index_global"]] = {"probs": {o: (conf if o == answer else rest) for o in ("good", "scratched", "unpainted")}, "latency_ms": 1.0}
            rungs.append((name, out))
        g = ladder_objective(decs, rungs, choose_thresholds(decs, rungs))
        j = ladder_objective(decs, rungs, choose_thresholds_joint(decs, rungs))
        assert j >= g - 1e-12, seed


def test_joint_per_question_fits_each_question_on_its_own():
    decs = decisions(12) + [{"index_global": 100 + i, "episode": f"ep{i % 4}", "question": "in_bin", "truth": "good"} for i in range(12)]
    lower = arm_answers(decs, 0.9, wrong_on=(3, 11), conf=0.8)  # wrong only on finish
    upper = arm_answers(decs, 1.0, conf=0.95)
    t = choose_thresholds_joint_per_question(decs, [("lower", lower), ("upper", upper)])
    assert set(t) == {"lower", "upper"} and set(t["lower"]) == {"finish", "in_bin"}
    assert t["lower"]["finish"] == 1.0
    assert ladder_objective([d for d in decs if d["question"] == "in_bin"], [("lower", lower), ("upper", upper)], {"lower": t["lower"]["in_bin"], "upper": t["upper"]["in_bin"]}) == 1.0


def test_per_question_fallback_uses_the_rung_threshold_for_thin_questions():
    # 30 finish decisions (fitted on their own) and 6 in_bin decisions (too few: take the rung's pooled threshold).
    decs = decisions(30) + [{"index_global": 200 + i, "episode": f"ep{i % 4}", "question": "in_bin", "truth": "good"} for i in range(6)]
    arm = arm_answers(decs, 1.0, conf=0.95)
    for i in range(200, 206):  # in_bin answers at 0.6 and all right: alone, they would earn a low cut-off
        arm[i]["probs"] = {"good": 0.6, "scratched": 0.2, "unpainted": 0.2}
    for i in range(10):  # ten finish answers wrong at 0.6: pooled, the 0.6 band loses, so the rung's cut-off is above it
        wrong = "unpainted" if decs[i]["truth"] == "good" else "good"
        arm[i]["probs"] = {o: (0.6 if o == wrong else 0.2) for o in ("good", "scratched", "unpainted")}
    rungs = [("arm", arm)]
    plain = choose_thresholds_per_question(decs, rungs)
    fallback = choose_thresholds_per_question(decs, rungs, min_decisions=10)
    pooled = choose_thresholds(decs, rungs)
    assert plain["arm"]["in_bin"] <= 0.6  # fitted on its own six
    assert pooled["arm"] > 0.6
    assert fallback["arm"]["in_bin"] == pooled["arm"]  # too few: the rung's single threshold, not its own
    assert fallback["arm"]["finish"] == plain["arm"]["finish"]  # enough decisions: unchanged
    assert set(fallback["arm"]) == {"finish", "in_bin"}


def test_ladder_ece_leaves_out_the_persons_rows():
    decs = decisions(10)
    answers = arm_answers(decs, 1.0, wrong_on={0, 1}, conf=0.95)  # 0.95 sure, 3 of 5 right: badly calibrated
    for i in range(5, 10):
        answers[i]["probs"] = {"good": 0.4, "scratched": 0.3, "unpainted": 0.3}  # under the threshold: the person answers
    rows = compose_ladder(decs, [("a", answers, 0.9)])
    assert sum(r["human"] for r in rows) == 5
    machine = [r for r in rows if not r["human"]]
    expected = expected_calibration_error([r["confidence"] for r in machine], [r["answer"] == r["truth"] for r in machine])
    assert abs(score("ladder", rows).ece - expected) < 1e-9 and expected > 0.3
