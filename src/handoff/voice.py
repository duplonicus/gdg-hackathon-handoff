"""Spoken answers for the operator page: audio to text with Whistle, text to one of the question's options with a model.

Speech never acts by itself. ``VoiceInput.hear`` only proposes an option; the page reads it back and the person
confirms it, and the buttons stay as the fallback. Whistle is a 16.9 MB speech model that runs on the CPU, so it takes
nothing from the GPU the Decider is using. It runs through its own native runner (``scripts/get_whistle.ps1``), not the
``cactus-needle`` Python package, whose calls send usage counts to its maker.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .decider import Question, RungAnswer

WHISTLE_EXE = Path(r"D:\Models\whistle\needle.exe")
WHISTLE_MODEL = Path(r"D:\Models\whistle\whistle.cact")
MAX_AUDIO_BYTES = 4_000_000  # 15 s at 48 kHz, 16-bit mono is about 1.5 MB
MAP_THRESHOLD = 0.80
UNCLEAR = "unclear"
DOMAIN_WORDS = ("rework", "good", "defective", "tray", "bin", "green", "orange", "abort", "grasp")


class Transcriber(Protocol):
    def transcribe(self, wav: bytes, keywords: Sequence[str] = ()) -> str: ...


class AnswerRung(Protocol):
    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer: ...


class Whistle:
    """Runs the Whistle runner on one clip. English is forced: the operator's short replies are too little to detect from."""

    def __init__(self, exe: Path = WHISTLE_EXE, model: Path = WHISTLE_MODEL, timeout_s: float = 30.0, run: Callable[..., Any] = subprocess.run) -> None:
        self.exe, self.model, self.timeout_s, self.run = Path(exe), Path(model), timeout_s, run

    def available(self) -> bool:
        return self.exe.is_file() and self.model.is_file()

    def transcribe(self, wav: bytes, keywords: Sequence[str] = ()) -> str:
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "clip.wav"
            clip.write_bytes(wav)
            command = [str(self.exe), "--model", str(self.model), "--audio", str(clip), "--audio-language", "en"]
            if keywords:
                words = Path(folder) / "keywords.txt"
                words.write_text("\n".join(keywords) + "\n", encoding="utf-8")
                command += ["--audio-keywords", str(words)]
            done = self.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout_s)
        if done.returncode != 0:
            raise RuntimeError(f"whistle exited {done.returncode}: {(done.stderr or done.stdout or '').strip()[:200]}")
        for line in reversed((done.stdout or "").splitlines()):
            if line.startswith("{"):
                return str(json.loads(line).get("text", "")).strip()
        raise RuntimeError(f"whistle printed no result: {(done.stdout or '').strip()[:200]}")


def keywords_for(question: Question) -> list[str]:
    """Words to favour: this question's options, then the words of the cell."""
    words = [option.replace("_", " ") for option in question.options]
    return words + [word for word in DOMAIN_WORDS if word not in words]


def choice_question(question: Question) -> Question:
    """The same options, asked about what the person meant, with a way to say they did not pick one."""
    criteria = {option: f"the operator means: {meaning}" for option, meaning in question.criteria.items()}
    criteria[UNCLEAR] = "the reply picks none of these, is cut off, contradicts itself, or asks something"
    return replace(question, text="Which option did the operator's reply choose?", options=(*question.options, UNCLEAR), criteria=criteria, safe_option=UNCLEAR)


def choice_state(question: Question, transcript: str) -> str:
    return f'The operator was asked: "{question.text}" The operator replied out loud, transcribed by a speech model: "{transcript}"'


def map_speech(question: Question, transcript: str, rung: AnswerRung, threshold: float = MAP_THRESHOLD) -> dict[str, Any]:
    """The option the transcript chose. ``option`` is None, with a ``reason``, unless the model is sure of a real option."""
    if not transcript.strip():
        return {"transcript": transcript, "option": None, "confidence": 0.0, "reason": "no speech heard"}
    answer = rung.answer(choice_question(question), choice_state(question, transcript), None)
    if answer.error or not answer.probs:
        return {"transcript": transcript, "option": None, "confidence": 0.0, "reason": f"the model did not answer: {answer.error}"}
    result = {"transcript": transcript, "option": None, "confidence": round(answer.confidence, 3)}
    if answer.top == UNCLEAR:
        return {**result, "reason": "that did not pick an option"}
    if answer.confidence < threshold:
        return {**result, "reason": f"not sure enough ({answer.top} at {answer.confidence:.2f})"}
    return {**result, "option": answer.top, "reason": ""}


class VoiceInput:
    def __init__(self, transcriber: Transcriber, rung: AnswerRung, threshold: float = MAP_THRESHOLD) -> None:
        self.transcriber, self.rung, self.threshold = transcriber, rung, threshold

    def hear(self, question: Question, wav: bytes) -> dict[str, Any]:
        try:
            transcript = self.transcriber.transcribe(wav, keywords_for(question))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            return {"transcript": "", "option": None, "confidence": 0.0, "reason": f"speech to text failed: {error}"}
        return map_speech(question, transcript, self.rung, self.threshold)


def is_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def voice_for_demo(rung: AnswerRung | None, say: Callable[[str], None] = print, whistle: Whistle | None = None) -> VoiceInput | None:
    """Voice for the operator page, or None with the reason printed. The buttons work either way."""
    whistle = whistle or Whistle()
    if rung is None:
        say("voice answers off: this ladder has no model to read what was said; buttons only")
        return None
    if not whistle.available():
        say(f"voice answers off: Whistle not found at {whistle.exe.parent} (run scripts/get_whistle.ps1); buttons only")
        return None
    return VoiceInput(whistle, rung)
