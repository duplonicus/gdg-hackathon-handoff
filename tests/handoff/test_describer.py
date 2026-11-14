"""Frame descriptions for the text arms (Nov 4, commit review round 5): one file per describer, and Qwen's describer
refuses to run unless Qwen is the model on its port."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import handoff.rungs as rungs

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts" / "benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_each_describer_has_its_own_file_and_the_default_keeps_the_old_name(tmp_path):
    assert bench.descriptions_path(tmp_path) == tmp_path / "descriptions.jsonl"
    assert bench.descriptions_path(tmp_path, "qwen") == tmp_path / "descriptions-qwen.jsonl"
    assert bench.descriptions_path(tmp_path, "gemma4:12b") == tmp_path / "descriptions-gemma4_12b.jsonl"


def test_qwen_describer_refuses_another_model_on_its_port(tmp_path, monkeypatch):
    row = {"episode": "ep0000", "index": 0, "question": "leak_block", "options": ["yes", "no"], "state_text": "", "frame": None, "truth": "no", "context": {}}
    (tmp_path / "decisions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(rungs, "served_model", lambda url: "d:\\models\\gguf\\d3-27b\\d3-27b-artemis-iq4_xs.gguf")
    monkeypatch.setattr(bench, "_post", lambda *a, **k: pytest.fail("posted to a server that is not Qwen"))
    with pytest.raises(RuntimeError):
        bench.describe(tmp_path, "qwen")
    assert not (tmp_path / "descriptions-qwen.jsonl").exists()


class FailingRung:
    name = "x"

    def answer(self, question, state_text, image_b64):
        from handoff.decider import RungAnswer, uniform

        return RungAnswer("x", uniform(question.options), 1.0, "ConnectionRefusedError: down")


def one_decision_log(tmp_path):
    row = {"episode": "ep0000", "index": 0, "question": "leak_block", "options": ["yes", "no"], "state_text": "s", "frame": None, "truth": "no", "context": {}}
    (tmp_path / "decisions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_a_replay_whose_calls_all_failed_exits_non_zero(tmp_path, monkeypatch):
    """Review round 6: a dead server made every call an error row and the replay still exited 0."""
    one_decision_log(tmp_path)
    monkeypatch.setattr(bench, "unload_ollama_models", lambda: None)
    monkeypatch.setattr(bench, "make_arm", lambda name, url: FailingRung())
    with pytest.raises(SystemExit) as stop:
        bench.replay(tmp_path, "d3-27b-vision", None, None)
    assert stop.value.code == 2


def test_a_text_arm_on_another_describer_is_recorded_as_its_own_arm(tmp_path, monkeypatch):
    one_decision_log(tmp_path)
    (tmp_path / "descriptions-qwen.jsonl").write_text(json.dumps({"index_global": 0, "model": "qwen", "description": "d"}) + "\n", encoding="utf-8")
    (tmp_path / "arms.jsonl").write_text(json.dumps({"arm": "decider-2b-text", "index_global": 0, "probs": {"yes": 0.5, "no": 0.5}, "error": None}) + "\n", encoding="utf-8")
    monkeypatch.setattr(bench, "unload_ollama_models", lambda: None)

    class Rung:
        name = "decider-2b-text"

        def answer(self, question, state_text, image_b64):
            from handoff.decider import RungAnswer

            assert "Camera: d" in state_text
            return RungAnswer("decider-2b-text", {"yes": 0.1, "no": 0.9}, 1.0)

    monkeypatch.setattr(bench, "make_arm", lambda name, url: Rung())
    bench.replay(tmp_path, "decider-2b-text", None, None, describer="qwen")
    arms = [json.loads(line)["arm"] for line in (tmp_path / "arms.jsonl").open(encoding="utf-8")]
    assert arms == ["decider-2b-text", "decider-2b-text@qwen"]  # the Gemma-described row did not stop the Qwen one
