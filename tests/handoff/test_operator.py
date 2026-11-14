"""The operator page: a question appears, an answer posted from a browser-like client is returned."""

import json
import threading
import urllib.request

from handoff.decider import QUESTIONS
from handoff.operator import HumanRung, OperatorPage


def test_operator_answer_round_trip():
    page = OperatorPage(port=0)
    page.start()
    try:
        idle_ticks = []

        def answer_from_browser():
            pending = {}
            for _ in range(200):
                pending = json.loads(urllib.request.urlopen(f"{page.url}api/pending").read())
                if pending.get("id") is not None:
                    break
            assert pending["options"] == list(QUESTIONS["finish"].options)
            assert pending["climbed"] == "decider-2b 0.55, gemma4:e4b 0.61"
            body = json.dumps({"id": pending["id"], "answer": "defective"}).encode()
            urllib.request.urlopen(urllib.request.Request(f"{page.url}api/answer", data=body, headers={"Content-Type": "application/json"}))

        threading.Thread(target=answer_from_browser, daemon=True).start()
        rung = HumanRung(page, idle=lambda: idle_ticks.append(1), timeout_s=10.0, climbed=lambda: "decider-2b 0.55, gemma4:e4b 0.61")
        a = rung.answer(QUESTIONS["finish"], "part under the camera", None)
        assert a.top == "defective" and a.confidence == 1.0 and a.error is None
        assert idle_ticks  # the robot kept stepping while it waited
        assert json.loads(urllib.request.urlopen(f"{page.url}api/pending").read()) == {}
    finally:
        page.close()


def test_operator_timeout_is_an_error_not_a_crash():
    page = OperatorPage(port=0)
    page.start()
    try:
        rung = HumanRung(page, idle=lambda: None, timeout_s=0.3)
        a = rung.answer(QUESTIONS["in_bin"], "", None)
        assert a.error and "operator" in a.error
        assert abs(sum(a.probs.values()) - 1.0) < 1e-9
    finally:
        page.close()


def test_the_film_truth_hint_rides_only_when_set_and_never_breaks_a_question():
    page = OperatorPage(port=0)
    page.start()
    try:
        seen = {}

        def read_and_answer():
            for _ in range(200):
                pending = json.loads(urllib.request.urlopen(f"{page.url}api/pending").read())
                if pending.get("id") is not None:
                    break
            seen.update(pending)
            body = json.dumps({"id": pending["id"], "answer": "good"}).encode()
            urllib.request.urlopen(urllib.request.Request(f"{page.url}api/answer", data=body))

        threading.Thread(target=read_and_answer, daemon=True).start()
        HumanRung(page, idle=lambda: None, timeout_s=10.0).answer(QUESTIONS["finish"], "part", None)
        assert "film_truth" not in seen  # an ordinary demo carries no hint

        def broken(question):
            raise RuntimeError("no part in view")

        page.truth_fn = broken
        seen.clear()
        threading.Thread(target=read_and_answer, daemon=True).start()
        a = HumanRung(page, idle=lambda: None, timeout_s=10.0).answer(QUESTIONS["finish"], "part", None)
        assert seen["film_truth"] is None and a.top == "good"

        page.truth_fn = lambda q: "defective"
        seen.clear()
        threading.Thread(target=read_and_answer, daemon=True).start()
        HumanRung(page, idle=lambda: None, timeout_s=10.0).answer(QUESTIONS["finish"], "part", None)
        assert seen["film_truth"] == "defective"
    finally:
        page.close()
