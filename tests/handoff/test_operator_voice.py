"""Spoken operator answers: speech only ever proposes, a person confirms, and every failure leaves the buttons working."""

import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from handoff.decider import QUESTIONS, RungAnswer
from handoff.operator import HumanRung, OperatorPage
from handoff.voice import (
    MAX_AUDIO_BYTES,
    UNCLEAR,
    VoiceInput,
    Whistle,
    choice_question,
    choice_state,
    is_wav,
    keywords_for,
    map_speech,
    voice_for_demo,
)

WAV = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40
QUESTION = QUESTIONS["in_bin"]


class FakeRung:
    def __init__(self, probs=None, error=None):
        self.probs, self.error, self.calls = probs if probs is not None else {"yes": 0.0, "no": 0.95, UNCLEAR: 0.05}, error, []

    def answer(self, question, state, image):
        self.calls.append((question, state, image))
        return RungAnswer("fake", self.probs, 1.0, self.error)


class FakeEars:
    def __init__(self, text="no, the wrong tray", error=None):
        self.text, self.error, self.calls = text, error, []

    def transcribe(self, wav, keywords=()):
        self.calls.append((wav, list(keywords)))
        if self.error:
            raise self.error
        return self.text


def post(url, data, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(request) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def get_pending(page):
    return json.loads(urllib.request.urlopen(f"{page.url}api/pending").read())


def wait_for_pending(page):
    for _ in range(2000):
        pending = get_pending(page)
        if pending.get("id") is not None:
            return pending
    raise AssertionError("no question appeared")


class Asking:
    """A page with a question open in a background thread, as the robot holds it."""

    def __init__(self, voice, timeout_s=10.0):
        self.page = OperatorPage(port=0, voice=voice)
        self.timeout_s, self.result = timeout_s, None

    def __enter__(self):
        self.page.start()
        rung = HumanRung(self.page, idle=lambda: None, timeout_s=self.timeout_s)
        self.thread = threading.Thread(target=lambda: setattr(self, "result", rung.answer(QUESTION, "state", None)), daemon=True)
        self.thread.start()
        self.pending = wait_for_pending(self.page)
        return self

    def __exit__(self, *exc):
        if self.thread.is_alive():
            post(f"{self.page.url}api/answer", json.dumps({"id": self.pending["id"], "answer": "no"}).encode())
        self.thread.join(5)
        self.page.close()

    def speak(self, wav=WAV, qid=None):
        return post(f"{self.page.url}api/voice?id={self.pending['id'] if qid is None else qid}", wav, {"Content-Type": "audio/wav"})

    def confirm(self, option, spoken=True):
        post(f"{self.page.url}api/answer", json.dumps({"id": self.pending["id"], "answer": option, "spoken": spoken}).encode())
        self.thread.join(5)
        return self.result


def test_speech_proposes_but_never_answers():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        status, heard = asking.speak()
        assert (status, heard["option"], heard["transcript"]) == (200, "no", "no, the wrong tray")
        assert asking.thread.is_alive()  # the robot is still holding
        assert get_pending(asking.page)["id"] == asking.pending["id"]  # and the question is still open


def test_confirmed_speech_answers_and_is_logged_with_what_was_said():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        asking.speak()
        result = asking.confirm("no")
    assert result.top == "no" and result.confidence == 1.0
    assert result.extra["spoken"]["transcript"] == "no, the wrong tray"
    assert result.extra["operator"] is True


def test_a_button_after_speech_is_not_logged_as_spoken():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        asking.speak()
        result = asking.confirm("yes", spoken=False)
    assert result.top == "yes"
    assert "spoken" not in result.extra


def test_a_spoken_flag_cannot_attach_speech_to_a_different_answer():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        asking.speak()  # proposed "no"
        result = asking.confirm("yes", spoken=True)
    assert result.top == "yes"
    assert "spoken" not in result.extra


def test_spoken_note_does_not_leak_into_the_next_question():
    page = OperatorPage(port=0, voice=VoiceInput(FakeEars(), FakeRung()))
    page.last_spoken = {"transcript": "old"}
    page._answers[0] = "no"  # the next question's answer is already in
    assert page.ask(QUESTION, "state", None, "", lambda: None, 1.0) == "no"
    assert page.last_spoken is None


def test_pending_says_whether_voice_is_on():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        assert asking.pending["voice"] is True
    with Asking(None) as asking:
        assert asking.pending["voice"] is False


@pytest.mark.parametrize(("wav", "qid_shift", "status"), [(b"not audio at all" * 4, 0, 400), (WAV, 5, 409)])
def test_bad_audio_and_stale_question_are_refused(wav, qid_shift, status):
    ears = FakeEars()
    with Asking(VoiceInput(ears, FakeRung())) as asking:
        got, reply = asking.speak(wav, asking.pending["id"] + qid_shift)
        assert got == status and "error" in reply
        assert ears.calls == []


def test_voice_off_refuses_audio():
    with Asking(None) as asking:
        assert asking.speak()[0] == 404


def test_oversized_audio_is_refused_from_its_header_without_reading_it():
    with Asking(VoiceInput(FakeEars(), FakeRung())) as asking:
        handler = asking.page._handler.__new__(asking.page._handler)
        sent = []
        handler._error = lambda status, message: sent.append(status)
        handler.rfile = SimpleNamespace(read=lambda n: pytest.fail("read an oversized body"))
        handler._voice(str(asking.pending["id"]), MAX_AUDIO_BYTES + 1)
        assert sent == [413]


def test_upload_limit_is_enough_for_the_page_longest_recording():
    fifteen_s_at_48k_16bit_mono = 44 + 15 * 48_000 * 2
    assert fifteen_s_at_48k_16bit_mono <= MAX_AUDIO_BYTES


@pytest.mark.parametrize(("probs", "error", "transcript", "option", "reason_has"), [
    ({"yes": 0.9, "no": 0.05, UNCLEAR: 0.05}, None, "yep", "yes", ""),
    ({"yes": 0.80, "no": 0.15, UNCLEAR: 0.05}, None, "yep", "yes", ""),
    ({"yes": 0.79, "no": 0.16, UNCLEAR: 0.05}, None, "maybe yes", None, "not sure"),
    ({"yes": 0.01, "no": 0.01, UNCLEAR: 0.98}, None, "what?", None, "did not pick"),
    ({}, "server down", "yes", None, "did not answer"),
    ({"yes": 0.5, "no": 0.5, UNCLEAR: 0.0}, "server down", "yes", None, "did not answer"),
])
def test_map_speech_only_proposes_a_sure_real_option(probs, error, transcript, option, reason_has):
    result = map_speech(QUESTION, transcript, FakeRung(probs, error))
    assert result["option"] == option
    assert reason_has in result["reason"]
    if option:
        assert result["reason"] == ""


def test_silence_does_not_reach_the_model():
    rung = FakeRung()
    assert map_speech(QUESTION, "   ", rung)["reason"] == "no speech heard"
    assert rung.calls == []


def test_model_is_asked_about_the_words_without_the_camera_frame():
    rung = FakeRung()
    map_speech(QUESTION, "send it back", rung)
    question, state, image = rung.calls[0]
    assert image is None
    assert '"send it back"' in state and QUESTION.text in state
    assert question.options == (*QUESTION.options, UNCLEAR)
    assert set(question.criteria) == set(question.options)
    assert question.id == QUESTION.id


def test_production_question_is_left_alone():
    before = (QUESTION.text, QUESTION.options, dict(QUESTION.criteria), QUESTION.safe_option)
    choice_question(QUESTION)
    choice_state(QUESTION, "x")
    assert before == (QUESTION.text, QUESTION.options, dict(QUESTION.criteria), QUESTION.safe_option)
    assert UNCLEAR not in QUESTION.options


@pytest.mark.parametrize("error", [RuntimeError("exit 1"), OSError("no exe"), subprocess.TimeoutExpired("whistle", 30)])
def test_transcription_failure_is_a_reason_not_a_crash(error):
    rung = FakeRung()
    heard = VoiceInput(FakeEars(error=error), rung).hear(QUESTION, WAV)
    assert heard["option"] is None and "speech to text failed" in heard["reason"]
    assert rung.calls == []


def test_hear_favours_this_questions_options():
    ears = FakeEars()
    VoiceInput(ears, FakeRung()).hear(QUESTIONS["safe_to_grasp"], WAV)
    words = ears.calls[0][1]
    assert words[:4] == ["grasp", "re look", "ask operator", "abort"]
    assert len(words) == len(set(words))


def test_keywords_have_no_duplicates_for_every_question():
    for question in QUESTIONS.values():
        words = keywords_for(question)
        assert len(words) == len(set(words)), question.id


def whistle_run(stdout='{"text":" Yes, it landed. ","language":"en"}\n', returncode=0, stderr=""):
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        clip = Path(command[command.index("--audio") + 1])
        seen["clip"] = clip.read_bytes()
        if "--audio-keywords" in command:
            seen["keywords"] = Path(command[command.index("--audio-keywords") + 1]).read_text(encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run, seen


def test_whistle_command_and_result():
    run, seen = whistle_run()
    text = Whistle(Path("w.exe"), Path("m.cact"), run=run).transcribe(WAV, ["yes", "no"])
    assert text == "Yes, it landed."
    assert seen["command"][:3] == ["w.exe", "--model", "m.cact"]
    assert seen["command"][seen["command"].index("--audio-language") + 1] == "en"
    assert seen["clip"] == WAV
    assert seen["keywords"] == "yes\nno\n"
    assert seen["kwargs"]["timeout"] == 30.0


def test_whistle_without_keywords_passes_no_keyword_file():
    run, seen = whistle_run()
    Whistle(Path("w.exe"), Path("m.cact"), run=run).transcribe(WAV)
    assert "--audio-keywords" not in seen["command"]


def test_whistle_reads_the_json_line_past_other_output():
    run, _ = whistle_run(stdout='loading\n{"text":"no"}\ndone\n')
    assert Whistle(Path("w"), Path("m"), run=run).transcribe(WAV) == "no"


@pytest.mark.parametrize(("kw", "message"), [
    ({"returncode": 3, "stdout": "", "stderr": "bad model"}, "exited 3: bad model"),
    ({"stdout": "nothing useful\n"}, "no result"),
])
def test_whistle_failures_raise(kw, message):
    run, _ = whistle_run(**kw)
    with pytest.raises(RuntimeError, match=message):
        Whistle(Path("w"), Path("m"), run=run).transcribe(WAV)


def test_is_wav():
    assert is_wav(WAV)
    assert not is_wav(b"RIFF")
    assert not is_wav(b"OggS" + b"\0" * 60)


def test_voice_for_demo_says_why_it_is_off(tmp_path):
    said = []
    missing = Whistle(tmp_path / "w.exe", tmp_path / "m.cact")
    assert voice_for_demo(None, said.append, missing) is None
    assert voice_for_demo(FakeRung(), said.append, missing) is None
    assert "no model" in said[0] and "get_whistle" in said[1]
    (tmp_path / "w.exe").write_bytes(b"x")
    (tmp_path / "m.cact").write_bytes(b"x")
    assert isinstance(voice_for_demo(FakeRung(), said.append, Whistle(tmp_path / "w.exe", tmp_path / "m.cact")), VoiceInput)
