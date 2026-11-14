"""The llama.cpp chat rung for Qwen3.8-27B (Nov 4): what it sends, and how it reads both kinds of answer."""

from __future__ import annotations

import json
import math

import pytest

import handoff.rungs as rungs
from handoff.decider import QUESTIONS
from handoff.rungs import LlamaChatRung


def test_first_word_reads_the_first_tokens_probabilities(monkeypatch):
    sent = {}

    def post(url, body, timeout):
        sent.update(url=url, body=body)
        top = [{"token": "leak", "logprob": math.log(0.7)}, {"token": "good", "logprob": math.log(0.2)}, {"token": "The", "logprob": math.log(0.1)}]
        return {"choices": [{"message": {"content": "leaker"}, "logprobs": {"content": [{"token": "leak", "top_logprobs": top}]}}]}

    monkeypatch.setattr(rungs, "_post", post)
    ans = LlamaChatRung("qwen-27b-vision", "http://x:1").answer(QUESTIONS["leak_capsule"], "state", "AAAA")
    assert ans.error is None
    assert ans.probs["leaker"] == pytest.approx(0.7 / 0.9) and ans.probs["good"] == pytest.approx(0.2 / 0.9)  # "The" starts no option
    assert sent["url"] == "http://x:1/v1/chat/completions"
    content = sent["body"]["messages"][0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False} and sent["body"]["logprobs"] is True


def test_self_mode_reads_the_stated_confidence(monkeypatch):
    def post(url, body, timeout):
        assert body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]
        return {"choices": [{"message": {"content": json.dumps({"answer": "no", "confidence": 0.9})}}]}

    monkeypatch.setattr(rungs, "_post", post)
    ans = LlamaChatRung("qwen-27b-self-vision", "http://x:1", mode="self").answer(QUESTIONS["leak_block"], "state", "AAAA")
    assert ans.probs == pytest.approx({"yes": 0.1, "no": 0.9})


def test_a_server_failure_is_a_logged_error_not_a_crash(monkeypatch):
    def post(url, body, timeout):
        raise ConnectionRefusedError("down")

    monkeypatch.setattr(rungs, "_post", post)
    ans = LlamaChatRung("qwen-27b-vision", "http://x:1").answer(QUESTIONS["tray_clear"], "state", None)
    assert ans.error and "down" in ans.error


def test_check_model_refuses_another_model(monkeypatch):
    monkeypatch.setattr(rungs, "served_model", lambda url: "d3-27b-artemis-iq4_xs.gguf")
    with pytest.raises(RuntimeError):
        LlamaChatRung("qwen-27b-vision", "http://x:1", expect=("qwen3.8-27b",)).check_model()
