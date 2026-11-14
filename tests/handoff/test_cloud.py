"""Cloud rungs: request shapes for the Gemini API and OpenRouter, stated-confidence parsing, keys, retries."""

import io
import json
import urllib.error

import pytest

import handoff.cloud as cloud
from handoff.cloud import GeminiRung, OpenRouterRung, api_key, post_json
from handoff.decider import QUESTIONS
from handoff.rungs import probs_from_self_confidence

FINISH = QUESTIONS["finish"]


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")


def test_stated_confidence_goes_to_the_answer_and_the_rest_is_spread():
    probs, extra = probs_from_self_confidence('{"answer": "good", "confidence": 0.8}', ("good", "defective"))
    assert probs == pytest.approx({"good": 0.8, "defective": 0.2})
    assert extra == {"self_confidence": 0.8}
    probs, _ = probs_from_self_confidence('{"answer": "no", "confidence": 1.7}', ("yes", "no", "unsure"))
    assert probs == {"yes": 0.0, "no": 1.0, "unsure": 0.0}


def test_an_answer_outside_the_options_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        probs_from_self_confidence('{"answer": "maybe", "confidence": 0.9}', ("good", "defective"))


def test_gemini_request_has_frame_first_json_schema_no_thinking_and_skips_thought_parts(monkeypatch):
    seen = {}

    def fake(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        parts = [{"text": "ignored", "thought": True}, {"text": json.dumps({"answer": FINISH.options[1], "confidence": 0.9})}]
        return {"candidates": [{"content": {"parts": parts}}], "usageMetadata": {"promptTokenCount": 1175, "candidatesTokenCount": 12}}

    monkeypatch.setattr(cloud, "post_json", fake)
    a = GeminiRung().answer(FINISH, "state", "aW1n")
    assert a.error is None and a.top == FINISH.options[1] and a.confidence == pytest.approx(0.9)
    assert seen["url"].endswith("/models/gemini-robotics-er-2-preview:generateContent")
    assert seen["headers"] == {"x-goog-api-key": "test-gemini"}
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0] == {"inline_data": {"mime_type": "image/png", "data": "aW1n"}} and "state" in parts[1]["text"]
    config = seen["body"]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingBudget": 0} and config["responseSchema"]["properties"]["answer"]["enum"] == list(FINISH.options)
    assert a.extra["tokens_in"] == 1175 and a.extra["tokens_out"] == 12


def test_openrouter_request_names_the_model_sends_a_data_url_and_records_cost(monkeypatch):
    seen = {}

    def fake(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": json.dumps({"answer": FINISH.options[0], "confidence": 0.6})}}], "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "cost": 0.0027}}

    monkeypatch.setattr(cloud, "post_json", fake)
    a = OpenRouterRung().answer(FINISH, "state", "aW1n")
    assert a.error is None and a.top == FINISH.options[0]
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions" and seen["body"]["model"] == "google/gemini-4-argon"
    assert seen["headers"] == {"Authorization": "Bearer test-openrouter"}
    content = seen["body"]["messages"][0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert a.extra["cost_usd"] == 0.0027


def test_no_frame_sends_text_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(cloud, "post_json", lambda url, body, headers, timeout: seen.update(body=body) or {"choices": [{"message": {"content": '{"answer": "good", "confidence": 0.5}'}}]})
    OpenRouterRung().answer(FINISH, "state", None)
    assert [c["type"] for c in seen["body"]["messages"][0]["content"]] == ["text"]


def test_key_from_dotenv_when_not_in_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY")
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENROUTER_API_KEY=\nGEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert api_key("GEMINI_API_KEY", env) == "from-file"
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        monkeypatch.delenv("OPENROUTER_API_KEY")
        api_key("OPENROUTER_API_KEY", env)


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"rate limited"))


def test_post_json_waits_out_rate_limits_then_returns(monkeypatch):
    calls, sleeps = [], []

    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": 1}'

    def fake_urlopen(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise http_error(429)
        return Ok()

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_urlopen)
    assert post_json("https://example.invalid/x", {}, {}, 1, sleep=sleeps.append) == {"ok": 1}
    assert len(calls) == 3 and sleeps == [5, 10]


def test_post_json_does_not_retry_a_bad_request_and_the_rung_logs_it(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda request, timeout: calls.append(1) or (_ for _ in ()).throw(http_error(400)))
    a = GeminiRung().answer(FINISH, "state", "aW1n")
    assert len(calls) == 1 and a.error is not None and "HTTP 400" in a.error
    assert a.probs == pytest.approx({o: 1 / len(FINISH.options) for o in FINISH.options})
