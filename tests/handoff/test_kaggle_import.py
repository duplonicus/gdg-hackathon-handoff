"""Kaggle Benchmarks run files become arms.jsonl rows in the same shape as the local arms."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("kaggle_import", Path(__file__).resolve().parents[2] / "scripts" / "kaggle_import.py")
kaggle_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kaggle_import)


def run_file(path: Path, *results: dict) -> None:
    path.write_text(json.dumps({"modelVersion": {"slug": "google/model-x"}, "results": [{"type": "AGGREGATED", "dictResult": r} for r in results]}), encoding="utf-8")


def result(index, answer, conf, truth="good", log="logA", question="finish"):
    return {"log": log, "index_global": float(index), "episode": "ep0001", "question": question, "answer": answer, "confidence": conf, "truth": truth}


def test_stated_confidence_becomes_probabilities_over_the_question_options():
    row = kaggle_import.row_from_result(result(3, "Defective", 0.9), "arm-x", "google/model-x")
    assert row["probs"] == pytest.approx({"good": 0.1, "defective": 0.9})
    assert row["index_global"] == 3 and row["error"] is None and row["truth"] == "good"
    four = kaggle_import.row_from_result(result(4, "abort", 0.92, truth="grasp", question="safe_to_grasp"), "arm-x", "m")
    assert set(four["probs"]) == {"grasp", "re_look", "ask_operator", "abort"} and four["probs"]["abort"] == pytest.approx(0.92)
    assert sum(four["probs"].values()) == pytest.approx(1.0)


def test_an_answer_outside_the_options_is_an_error_row_with_uniform_probs():
    row = kaggle_import.row_from_result(result(5, "maybe", 0.8), "arm-x", "m")
    assert row["error"] and row["probs"] == {"good": 0.5, "defective": 0.5}


def test_import_writes_each_log_once_and_a_rerun_adds_nothing(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    run_file(runs / "a.run.json", result(0, "good", 0.99))
    run_file(runs / "b.run.json", result(7, "defective", 0.6, truth="defective", log="logB"))
    sort = tmp_path / "sort"
    (sort / "logA").mkdir(parents=True)
    (sort / "logB").mkdir()
    (sort / "logA" / "arms.jsonl").write_text(json.dumps({"arm": "other", "index_global": 0}) + "\n", encoding="utf-8")
    args = ["kaggle_import.py", "--runs", str(runs), "--arm", "arm-x", "--sort-dir", str(sort)]
    monkeypatch.setattr("sys.argv", args)
    kaggle_import.main()
    kaggle_import.main()
    a = [json.loads(line) for line in (sort / "logA" / "arms.jsonl").open()]
    b = [json.loads(line) for line in (sort / "logB" / "arms.jsonl").open()]
    assert [r["arm"] for r in a] == ["other", "arm-x"] and [r["index_global"] for r in b] == [7]
