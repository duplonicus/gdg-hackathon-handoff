"""The run status a person watches: model servers up or down, and the run's log."""

import handoff.status as status_module
from handoff.decider import Decision, RungAnswer
from handoff.status import RunStatus, bump_rows, check_decider, check_gemma, describe, finish_line, rung_text


def decision():
    rungs = [
        RungAnswer("decider-2b", {"good": 0.5, "defective": 0.5}, 2040.0, error="URLError: refused"),
        RungAnswer("gemma4:e4b", {"good": 0.13, "defective": 0.87}, 330.0),
        RungAnswer("human", {"good": 0.0, "defective": 1.0}, 67000.0),
    ]
    return Decision("finish", ("good", "defective"), "state", None, rungs, "defective", "human", {}, 6.1, context={"part": "part_03"})


def test_describe_shows_an_error_as_an_error_not_as_a_confidence():
    line = describe(decision())
    assert line == "finish part_03: decider-2b ERROR -> gemma4:e4b defective 0.87 -> human defective 1.00 => defective (by human)"


def test_a_server_that_does_not_answer_is_reported_down(monkeypatch):
    def refuse(url, timeout=3.0):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(status_module, "_get", refuse)
    assert check_decider()["ok"] is False and "not running" in check_decider()["detail"]
    assert check_gemma()["ok"] is False


def test_gemma_up_but_not_in_memory_says_so(monkeypatch):
    monkeypatch.setattr(status_module, "_get", lambda url, timeout=3.0: {"models": []})
    assert check_gemma() == {"name": "Gemma", "ok": True, "detail": "Ollama is up; gemma4:e4b loads on its first question"}
    monkeypatch.setattr(status_module, "_get", lambda url, timeout=3.0: {"models": [{"name": "gemma4:e4b"}]})
    assert check_gemma()["detail"] == "loaded: gemma4:e4b"


def test_run_status_keeps_the_log_writes_it_to_disk_and_notes_a_model_going_down(tmp_path, monkeypatch):
    up = {"status": "ok", "model": "decider-2B"}
    answers = {"decider": up}

    def get(url, timeout=3.0):
        if url.endswith("/health"):
            if answers["decider"] is None:
                raise ConnectionRefusedError("down")
            return answers["decider"]
        return {"models": [{"name": "gemma4:e4b"}]}

    monkeypatch.setattr(status_module, "_get", get)
    status = RunStatus("local", tmp_path / "run.log")
    status.decision(decision())
    snap = status.snapshot()
    assert [b["ok"] for b in snap["badges"]] == [True, True] and snap["title"] == "Sorting run (local ladder)"
    assert snap["lines"][-1].endswith("=> defective (by human)")

    answers["decider"] = None
    status.check_models()
    snap = status.snapshot()
    assert snap["badges"][0]["ok"] is False
    assert "Decider is now DOWN" in snap["lines"][-1]
    models, last = status.caption_lines(1)
    assert models == "Decider: DOWN   Gemma: UP" and len(last) == 1
    status.close()
    assert "Decider is now DOWN" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_oracle_ladder_reports_no_models_instead_of_checking_servers(monkeypatch):
    def never(url, timeout=3.0):
        raise AssertionError("the oracle ladder must not ask the model servers")

    monkeypatch.setattr(status_module, "_get", never)
    badge = RunStatus("oracle").snapshot()["badges"][0]
    assert badge["state"] == "NOT USED" and badge["detail"].startswith("oracle")
    assert RunStatus("oracle").caption_lines()[0] == "Models: NOT USED"


def test_an_unclear_scale_reading_reads_unclear_not_a_half_sure_yes():
    rungs = [RungAnswer("scale", {"yes": 0.5, "no": 0.5}, 0.0, None, {"where": "unclear"}), RungAnswer("d3-27b-vision", {"yes": 0.9, "no": 0.1}, 900.0)]
    d = Decision("in_bin", ("yes", "no"), "state", None, rungs, "yes", "d3-27b-vision", {}, 9.0, context={"part": "part_02"})
    assert describe(d) == "in_bin part_02: scale unclear -> d3-27b-vision yes 0.90 => yes (by d3-27b-vision)"
    assert rung_text(RungAnswer("scale", {"yes": 1.0, "no": 0.0}, 0.0)) == "scale yes 1.00"


def test_sensor_windows_keep_first_set_order_and_replace_their_rows():
    status = RunStatus("oracle")
    assert status.snapshot()["sensors"] == []
    status.set_sensor("Tray load cells", [{"label": "good tray", "value": "0.0 g", "detail": "0 parts", "state": "ok"}])
    status.set_sensor("Station", [{"label": "part", "value": "none", "detail": "", "state": "idle"}])
    rows = [{"label": "good tray", "value": "51.0 g", "detail": "1 part", "state": "ok"}]
    status.set_sensor("Tray load cells", rows)
    rows[0]["value"] = "changed after set"  # the window keeps its own copy
    sensors = status.snapshot()["sensors"]
    assert [w["name"] for w in sensors] == ["Tray load cells", "Station"]
    assert sensors[0]["rows"] == [{"label": "good tray", "value": "51.0 g", "detail": "1 part", "state": "ok"}]
    assert len(sensors[0]["updated"]) == 8 and sensors[0]["updated"].count(":") == 2  # HH:MM:SS
    sensors[0]["rows"].clear()  # nor can a reader of the snapshot change it
    assert status.snapshot()["sensors"][0]["rows"]


def finish(part, answer, truth_, by="human"):
    return Decision("finish", ("good", "defective"), "", None, [], answer, by, {}, 1.0, truth=truth_, context={"part": part})


def test_the_finish_line_does_not_call_a_mis_sort_a_success():
    checks = {"parts_in_correct_bin": 2, "n_parts": 3, "mis_sorted": {"part_10": "in the good bin, belongs in rework"}, "bumps": 1}
    decisions = [finish("part_03", "good", "good"), finish("part_10", "good", "defective")]
    line = finish_line("success", checks, decisions)
    assert line.startswith("run finished: completed, 1 of 3 parts mis-sorted (finish part_10: answered good by human, truth defective); bumps 1; checks ")
    # a part that is in the wrong place for another reason than its finish answer says where it is
    held = finish_line("failure", {**checks, "mis_sorted": {"part_03": "held back to the reject lane"}}, decisions)
    assert held.startswith("run finished: failure, 1 of 3 parts mis-sorted (part_03: held back to the reject lane)")
    ok = finish_line("success", {"parts_in_correct_bin": 3, "n_parts": 3, "mis_sorted": {}, "bumps": 0}, decisions)
    assert ok.startswith("run finished: success, 3 of 3 parts in the right bin; bumps 0")


def test_a_wrong_finish_answer_shows_in_the_checks_and_the_finish_line(tmp_path):
    """End to end on the fake simulator: the first part's finish answer is flipped, the sort still completes."""
    import handoff.tasks.sort as sort_module
    from handoff.decider import Ladder, Step
    from handoff.logbook import Logbook
    from handoff.rungs import OracleRung
    from handoff.scene import Layout, make_episode, make_scene, place_episode
    from handoff.sim.fake import FakeSimulator
    from handoff.skills import TRAVEL_POSE

    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 0)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    first = episode.parts[0]

    def answer(q):
        t = sort_module.truth(sim, holder["ctx"], holder["part"], q.id)
        if q.id == "finish" and holder["part"] == first:
            return "defective" if t == "good" else "good"
        return t

    ctx = sort_module.SortContext(layout=layout, episode=episode, ladder=Ladder([Step(OracleRung(answer), 0.0)]), logbook=Logbook(tmp_path, "ep0"))
    holder["ctx"] = ctx
    real_ask = sort_module.ask

    def tracking_ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return real_ask(sim_, ctx_, qid, part)

    sort_module.ask = tracking_ask
    try:
        result = sort_module.run_sort(sim, ctx)
    finally:
        sort_module.ask = real_ask
    assert result.status.value == "success" and result.checks["parts_in_correct_bin"] == 2
    right = "rework" if episode.truth(first).defective else "good"
    wrong = "good" if right == "rework" else "rework"
    assert result.checks["mis_sorted"] == {first: f"in the {wrong} bin, belongs in {right}"}
    assert result.checks["bumps"] == len(getattr(ctx, "bumps", []))
    line = finish_line(result.status.value, result.checks, result.decisions)
    by = next(d.answered_by for d in result.decisions if d.question_id == "finish" and d.context["part"] == first)
    assert f"completed, 1 of 3 parts mis-sorted (finish {first}: answered {'defective' if right == 'good' else 'good'} by {by}, truth {'good' if right == 'good' else 'defective'})" in line


def test_bump_rows_show_the_raised_crossing_and_the_newest_bump_first():
    assert bump_rows([])[0] == {"label": "crossing extra", "value": "+0 cm", "detail": "0 bumps so far", "state": "ok"}
    bumps = [
        {"box": "good_wall_ym", "skill": "move_to_bin", "wrist_z": 1.0312, "retry_z": 1.08, "clearance_extra": 0.03},
        {"box": "rework_wall_xm", "skill": "lower", "wrist_z": 0.9714, "retry_z": None, "clearance_extra": 0.06},
    ]
    rows = bump_rows(bumps)
    assert rows[0]["value"] == "+6 cm" and rows[0]["detail"] == "2 bumps so far" and rows[0]["state"] == "warn"
    assert [r["label"] for r in rows[1:]] == ["rework_wall_xm", "good_wall_ym"]
    assert rows[1]["detail"] == "lower: let go there" and rows[2]["detail"] == "move_to_bin: retried at 1.080 m"
    assert all(set(r) == {"label", "value", "detail", "state"} for r in rows)  # the sensor-window row shape (C1)
