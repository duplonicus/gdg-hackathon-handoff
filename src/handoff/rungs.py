"""Rung implementations: the things a ladder can ask.

- :class:`DeciderRung`: a Strands Decider served locally (``strands-decider serve``), which
  returns a calibrated probability per option, with or without the camera frame.
- :class:`GemmaRung`: a Gemma 4 model through Ollama, answering with one word; the option
  probabilities are read from the first answer token's top log-probabilities.
- :class:`GemmaSelfRung`: Gemma answering in JSON with a confidence it reports about itself.
  This is the "small Gemma hands off to a bigger Gemma by its own judgement" baseline.
- :class:`OracleRung`: answers from the sim's truth with a fixed delay; stands in for a person
  during batch runs so hundreds of decisions can be logged unattended.
- :class:`HumanRung` lives in ``operator.py`` (it needs the web page).

All calls use the standard library only (the kit's environment has no requests library).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from .decider import Question, RungAnswer, normalised, timed

# 127.0.0.1, not localhost: on Windows the name resolves to ::1 first, where the WSL-side Ollama never answers.
# HANDOFF_OLLAMA_URL / HANDOFF_DECIDER_URL override them (the native Windows Ollama serves on 11435 while the
# Docker one still owns 11434).
OLLAMA_URL = os.environ.get("HANDOFF_OLLAMA_URL", "http://127.0.0.1:11434")
DECIDER_URL = os.environ.get("HANDOFF_DECIDER_URL", "http://127.0.0.1:8765")


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def served_model(url: str) -> str:
    """What a decision server says it has loaded, lower-cased: /health "model", every /v1/models name or id, and
    /props model_path (llama-server), joined; "" when it says nothing. Several arms share one port, so a replay checks
    this before answering under an arm's name."""
    seen: list[str] = []
    for path in ("/health", "/v1/models", "/props"):
        try:
            body = _get(f"{url}{path}")
        except Exception:  # noqa: BLE001, S112 - a server without this route says nothing here
            continue
        if not isinstance(body, dict):
            continue
        for key in ("model", "model_path"):
            if isinstance(body.get(key), str):
                seen.append(body[key])
        for m in [*(body.get("data") or []), *(body.get("models") or [])]:
            if isinstance(m, dict):
                seen += [m[k] for k in ("id", "name", "model") if isinstance(m.get(k), str)]
    return " ".join(dict.fromkeys(seen)).lower()


def unload_ollama_models(url: str = OLLAMA_URL) -> None:
    """Unload whatever Ollama has resident before an arm loads its own model.

    A loaded Gemma holds about 8 GB of the container's system RAM even when it reports 100% GPU; two of them
    (keep_alive 30 min) took the 24 GB WSL VM to zero free memory on Oct 13 and the replay chain was killed.
    """
    try:
        loaded = [m["name"] for m in _get(f"{url}/api/ps").get("models", [])]
        for name in loaded:
            _post(f"{url}/api/generate", {"model": name, "keep_alive": 0}, 60)
            print(f"unloaded {name} from Ollama")
    except Exception as error:  # noqa: BLE001 - Ollama down or not involved: the arm's own call will say so
        print(f"could not unload Ollama models: {error}")


class DeciderRung:
    """Any server that speaks the TypeSafe ``/v1/systemone`` shape: the Strands Decider server, or Ollama
    (0.35 and later) with ``model`` naming one of its decision models (clef-flash, nimble, tev1, laya, ...)."""

    def __init__(self, name: str = "decider-2b", url: str = DECIDER_URL, *, with_image: bool = True, timeout: float = 60.0, model: str | None = None, data_url: bool = False, image_field: str = "images", expect: tuple[str, ...] = ()) -> None:
        # data_url: send images as "data:image/png;base64,..." (d3's server refuses bare base64)
        # image_field: where the server reads the frame. "images" (TypeSafe shape, most servers); "winnow" (EldanRing's
        # Winnow: body["winnow"]["images"]); "state" (llm2jev / JPT: state becomes one user message holding the image
        # and the text); "media" (basal: body["media"] = [{"type": "image", "data": ...}], prompt language "en");
        # "qev" (OneJev's qev engine: the state names the frame as <image:1> and body["media"] carries it).
        # expect: words the served model's name must hold (served_model); arms that share a port set it.
        self.expect = tuple(w.lower() for w in expect)
        self.name, self.url, self.with_image, self.timeout, self.model, self.data_url = name, url, with_image, timeout, model, data_url
        self.image_field = image_field

    def healthy(self) -> bool:
        try:
            if self.model is not None:  # Ollama: the model has to be pulled
                return any(m.get("name", "").split(":")[0] == self.model.split(":")[0] for m in _get(f"{self.url}/api/tags").get("models", []))
            return _get(f"{self.url}/health").get("status") in ("ok", "ready")  # d3's server says "ready"
        except Exception:  # noqa: BLE001
            return False

    def check_model(self) -> str:
        """The served model's name; raises when the arm names words it must hold and the server does not have them."""
        served = served_model(self.url)
        if self.expect and not all(w in served for w in self.expect):
            raise RuntimeError(f"{self.name}: {self.url} serves {served or 'an unnamed model'!r}, expected {' + '.join(self.expect)}")
        return served

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            body: dict[str, Any] = {
                "state": state_text,
                "questions": {question.id: {"type": "choice", "instructions": question.text, "criteria": dict(question.criteria)}},
            }
            if self.model is not None:
                body["model"] = self.model
            if self.with_image and image_b64:
                image = f"data:image/png;base64,{image_b64}" if self.data_url or self.image_field != "images" else image_b64
                if self.image_field == "winnow":
                    body["winnow"] = {"images": [image]}
                elif self.image_field == "state":
                    body["state"] = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": state_text}]}]
                elif self.image_field == "qev":
                    body["state"] = {"camera": "<image:1>", "scene": state_text}
                    body["media"] = [{"type": "image", "data": image}]
                elif self.image_field == "media":
                    body["media"] = [{"type": "image", "data": image}]
                    body["lang"] = "en"
                else:
                    body["images"] = [image]
            reply = _post(f"{self.url}/v1/systemone", body, self.timeout)
            ans = reply["answers"][question.id]
            return ans["probabilities"], {"server_latency_ms": reply.get("latency_ms"), "with_image": bool(self.with_image and image_b64)}

        return timed(self.name, question.options, call)


def one_word_prompt(question: Question, state_text: str) -> str:
    options = ", ".join(f"{o} ({question.criteria[o]})" for o in question.options)
    words = ", ".join(question.options[:-1]) + f" or {question.options[-1]}"
    return (
        f"{state_text}\n\n{question.text}\nThe options are: {options}.\n"
        f"Answer with exactly one word: {words}."
    )


def probs_from_top_logprobs(top: list[dict[str, Any]], options: tuple[str, ...]) -> dict[str, float]:
    """Sum the first token's probability mass over tokens that start an option (case-insensitive).

    A token such as "def" counts for "defective"; "good" and "Good" for "good".
    """
    mass = {o: 0.0 for o in options}
    for entry in top:
        token = str(entry.get("token", "")).strip().lower()
        if not token:
            continue
        p = math.exp(float(entry.get("logprob", -math.inf)))
        for o in options:
            if o.lower().startswith(token) or token.startswith(o.lower()):
                mass[o] += p
                break
    return normalised(mass, options)


class GemmaRung:
    """One-word answer; probabilities from the first token's top log-probabilities."""

    def __init__(self, model: str = "gemma4:e4b", name: str | None = None, url: str = OLLAMA_URL, *, with_image: bool = True, timeout: float = 600.0) -> None:
        self.model, self.name, self.url, self.with_image, self.timeout = model, name or model, url, with_image, timeout

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            message: dict[str, Any] = {"role": "user", "content": one_word_prompt(question, state_text)}
            if self.with_image and image_b64:
                message["images"] = [image_b64]
            body = {
                "model": self.model,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "logprobs": True,
                "top_logprobs": 10,
                "options": {"temperature": 0, "num_predict": 4},
                "messages": [message],
            }
            reply = _post(f"{self.url}/api/chat", body, self.timeout)
            logprobs = reply.get("logprobs") or reply.get("message", {}).get("logprobs") or []
            top = logprobs[0].get("top_logprobs", []) if logprobs else []
            text = reply.get("message", {}).get("content", "").strip()
            probs = probs_from_top_logprobs(top, question.options) if top else _one_hot_from_text(text, question.options)
            return probs, {"text": text[:40], "eval_ms": (reply.get("eval_duration") or 0) // 1_000_000, "with_image": bool(self.with_image and image_b64)}

        return timed(self.name, question.options, call)


class LlamaChatRung:
    """A vision-language model behind llama.cpp's OpenAI-style chat endpoint (Qwen3.8-27B, the model d3 27B is built
    on: dup, Nov 4). ``mode="first_word"``: a one-word answer, probabilities from the first token's top log-probabilities,
    as GemmaRung reads Gemma. ``mode="self"``: a JSON answer with a confidence the model states, as GemmaSelfRung.
    Thinking is switched off: the answer must be the first token."""

    def __init__(self, name: str, url: str, *, mode: str = "first_word", with_image: bool = True, timeout: float = 120.0, expect: tuple[str, ...] = ()) -> None:
        if mode not in ("first_word", "self"):
            raise ValueError(f"mode {mode!r}")
        self.name, self.url, self.mode, self.with_image, self.timeout, self.expect = name, url, mode, with_image, timeout, expect

    def check_model(self) -> str:
        served = served_model(self.url)
        missing = [word for word in self.expect if word not in served]
        if missing:
            raise RuntimeError(f"{self.url} serves {served!r}, not {self.name} (missing {missing})")
        return served

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            prompt = one_word_prompt(question, state_text) if self.mode == "first_word" else self_confidence_prompt(question, state_text)
            content: list[dict[str, Any]] = []
            if self.with_image and image_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            content.append({"type": "text", "text": prompt})
            body: dict[str, Any] = {"messages": [{"role": "user", "content": content}], "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            if self.mode == "first_word":
                body.update({"max_tokens": 4, "logprobs": True, "top_logprobs": 10})
            else:
                body.update({"max_tokens": 64, "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": self_confidence_schema(question)}}})
            reply = _post(f"{self.url}/v1/chat/completions", body, self.timeout)
            choice = reply["choices"][0]
            text = (choice.get("message", {}).get("content") or "").strip()
            if self.mode == "self":
                probs, extra = probs_from_self_confidence(text, question.options)
                return probs, {**extra, "with_image": bool(self.with_image and image_b64)}
            tokens = (choice.get("logprobs") or {}).get("content") or []
            top = tokens[0].get("top_logprobs", []) if tokens else []
            probs = probs_from_top_logprobs(top, question.options) if top else _one_hot_from_text(text, question.options)
            return probs, {"text": text[:40], "with_image": bool(self.with_image and image_b64)}

        return timed(self.name, question.options, call)


def _one_hot_from_text(text: str, options: tuple[str, ...]) -> dict[str, float]:
    low = text.lower()
    for o in options:
        if low.startswith(o.lower()):
            return {x: (1.0 if x == o else 0.0) for x in options}
    return {o: 1.0 / len(options) for o in options}


def self_confidence_prompt(question: Question, state_text: str) -> str:
    options = ", ".join(f"{o} ({question.criteria[o]})" for o in question.options)
    return (
        f"{state_text}\n\n{question.text}\nThe options are: {options}.\n"
        "Reply in JSON with your answer and how confident you are from 0 to 1."
    )


def self_confidence_schema(question: Question) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(question.options)}, "confidence": {"type": "number"}},
        "required": ["answer", "confidence"],
    }


def probs_from_self_confidence(content: str, options: tuple[str, ...]) -> tuple[dict[str, float], dict[str, Any]]:
    """The stated confidence goes to the chosen option and the rest is spread evenly over the others."""
    parsed = json.loads(content)
    answer, conf = parsed["answer"], float(min(max(parsed.get("confidence", 0.5), 0.0), 1.0))
    if answer not in options:
        raise ValueError(f"answer {answer!r} is not an option")
    rest = (1.0 - conf) / max(len(options) - 1, 1)
    return {o: (conf if o == answer else rest) for o in options}, {"self_confidence": conf}


class GemmaSelfRung:
    """JSON answer plus a confidence the model states about itself (0 to 1)."""

    def __init__(self, model: str = "gemma4:e2b", name: str | None = None, url: str = OLLAMA_URL, *, with_image: bool = True, timeout: float = 600.0) -> None:
        self.model, self.name, self.url, self.with_image, self.timeout = model, name or f"{model}-self", url, with_image, timeout

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            message: dict[str, Any] = {"role": "user", "content": self_confidence_prompt(question, state_text)}
            if self.with_image and image_b64:
                message["images"] = [image_b64]
            body = {"model": self.model, "stream": False, "think": False, "keep_alive": "30m", "format": self_confidence_schema(question), "options": {"temperature": 0}, "messages": [message]}
            reply = _post(f"{self.url}/api/chat", body, self.timeout)
            return probs_from_self_confidence(reply["message"]["content"], question.options)

        return timed(self.name, question.options, call)


class OracleRung:
    """The sim's truth, as a stand-in person: always right, after a fixed delay."""

    def __init__(self, truth: Callable[[Question], str | None], name: str = "oracle", delay_s: float = 0.0) -> None:
        self.truth, self.name, self.delay_s = truth, name, delay_s

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            if self.delay_s:
                time.sleep(self.delay_s)
            answer = self.truth(question)
            if answer is None:
                raise ValueError("no truth available for this question")
            return {o: (1.0 if o == answer else 0.0) for o in question.options}, {}

        return timed(self.name, question.options, call)


class ReplayRung:
    """Serves a recorded answer for a (question, state, frame) key; for the benchmark replay."""

    def __init__(self, name: str, lookup: Callable[[str, str, str | None], dict[str, float] | None]) -> None:
        self.name, self.lookup = name, lookup

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            probs = self.lookup(question.id, state_text, image_b64)
            if probs is None:
                raise KeyError("no recorded answer")
            return probs, {"replayed": True}

        return timed(self.name, question.options, call)
