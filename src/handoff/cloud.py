"""Cloud rungs: frontier models behind an API key, the ceiling tier of the benchmark.

- :class:`GeminiRung`: Google's Gemini API (``generateContent``). Default model is Gemini Robotics ER 2
  (``gemini-robotics-er-2-preview``), Google's embodied-reasoning model; free tier.
- :class:`OpenRouterRung`: any OpenRouter chat model (default ``google/gemini-4-argon``, paid).

Neither API returns log-probabilities for these models (Google answers 400 "Logprobs is not enabled for this
model"; OpenRouter lists no logprobs parameter for Argon), so both answer in JSON with a confidence they state
about themselves, the same method as :class:`~handoff.rungs.GemmaSelfRung`.

Keys come from the environment or the repo's gitignored ``.env`` (``GEMINI_API_KEY``, ``OPENROUTER_API_KEY``).
Standard library only, like ``rungs.py``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .decider import Question, RungAnswer, timed
from .rungs import probs_from_self_confidence, self_confidence_prompt, self_confidence_schema

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
RETRY_STATUS = {429, 500, 502, 503, 504}


def api_key(name: str, env_file: Path = ENV_FILE) -> str:
    """The key from the environment, else from ``.env``; never printed."""
    if os.environ.get(name):
        return os.environ[name]
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key == name and not key.startswith("#") and value.strip():
                return value.strip()
    raise RuntimeError(f"{name} is not set (environment or {env_file.name})")


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float, *, retries: int = 6, sleep=time.sleep) -> dict[str, Any]:
    """POST with backoff on rate limits and server errors, so a replay waits instead of logging failures."""
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == retries:
                raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:200]}") from None
            sleep(min(5 * 2**attempt, 120))
    raise AssertionError("unreachable")


class GeminiRung:
    def __init__(self, model: str = "gemini-robotics-er-2-preview", name: str | None = None, *, with_image: bool = True, thinking_budget: int = 0, timeout: float = 120.0) -> None:
        self.model, self.name, self.with_image, self.thinking_budget, self.timeout = model, name or model, with_image, thinking_budget, timeout
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            parts: list[dict[str, Any]] = []
            if self.with_image and image_b64:
                parts.append({"inline_data": {"mime_type": "image/png", "data": image_b64}})
            parts.append({"text": self_confidence_prompt(question, state_text)})
            body = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": self_confidence_schema(question),
                    "thinkingConfig": {"thinkingBudget": self.thinking_budget},
                },
            }
            reply = post_json(self.url, body, {"x-goog-api-key": api_key("GEMINI_API_KEY")}, self.timeout)
            text = "".join(p.get("text", "") for p in reply["candidates"][0]["content"]["parts"] if not p.get("thought"))
            probs, extra = probs_from_self_confidence(text, question.options)
            usage = reply.get("usageMetadata", {})
            return probs, {**extra, "tokens_in": usage.get("promptTokenCount"), "tokens_out": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0), "with_image": bool(self.with_image and image_b64)}

        return timed(self.name, question.options, call)


class OpenRouterRung:
    def __init__(self, model: str = "google/gemini-4-argon", name: str | None = None, *, with_image: bool = True, reasoning_effort: str | None = "low", timeout: float = 180.0) -> None:
        self.model, self.name, self.with_image, self.reasoning_effort, self.timeout = model, name or model, with_image, reasoning_effort, timeout
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        def call() -> tuple[dict[str, float], dict[str, Any]]:
            content: list[dict[str, Any]] = [{"type": "text", "text": self_confidence_prompt(question, state_text)}]
            if self.with_image and image_b64:
                content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            body: dict[str, Any] = {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {**self_confidence_schema(question), "additionalProperties": False}}},
                "usage": {"include": True},
            }
            if self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort}
            reply = post_json(self.url, body, {"Authorization": f"Bearer {api_key('OPENROUTER_API_KEY')}"}, self.timeout)
            probs, extra = probs_from_self_confidence(reply["choices"][0]["message"]["content"], question.options)
            usage = reply.get("usage", {})
            return probs, {**extra, "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"), "cost_usd": usage.get("cost"), "with_image": bool(self.with_image and image_b64)}

        return timed(self.name, question.options, call)
