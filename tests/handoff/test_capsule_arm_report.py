"""capsule_arm_report.py (Nov 7): the per-arm rates the capsule prereg reports, counted the right way round."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("capsule_arm_report", ROOT / "scripts" / "capsule_arm_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def write_log(tmp_path: Path, decisions: list[dict], arms: list[dict]) -> Path:
    (tmp_path / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n", encoding="utf-8")
    (tmp_path / "arms.jsonl").write_text("\n".join(json.dumps(a) for a in arms) + "\n", encoding="utf-8")
    return tmp_path


def test_rates_on_the_leaker_calls(tmp_path):
    decisions = [
        {"episode": "ep1", "question": "leak_capsule", "truth": "leaker"},  # caught
        {"episode": "ep1", "question": "leak_capsule", "truth": "leaker"},  # missed at 0.95: confidently missed
        {"episode": "ep2", "question": "leak_block", "truth": "no"},  # false alarm
        {"episode": "ep2", "question": "leak_block", "truth": "no"},  # right
        {"episode": "ep2", "question": "where_needed", "truth": "left"},  # not a leaker call, wrong at 0.92
        {"episode": "ep2", "question": "tray_clear", "truth": None},  # unlabelled: left out
    ]
    arms = [
        {"arm": "x", "index_global": 0, "probs": {"leaker": 0.8, "good": 0.2}},
        {"arm": "x", "index_global": 1, "probs": {"leaker": 0.05, "good": 0.95}},
        {"arm": "x", "index_global": 2, "probs": {"yes": 0.6, "no": 0.4}},
        {"arm": "x", "index_global": 3, "probs": {"yes": 0.1, "no": 0.9}},
        {"arm": "x", "index_global": 4, "probs": {"left": 0.08, "right": 0.92}},
        {"arm": "x", "index_global": 5, "probs": {"yes": 0.9, "no": 0.1}},
    ]
    rows = rep.rows_for(write_log(tmp_path, decisions, arms))["x"]
    r = rep.rates(rep.counts(rows))
    assert len(rows) == 5
    assert r["caught"] == 0.5 and r["sure_missed"] == 0.5 and r["false_alarms"] == 0.5
    assert r["wrong_all"] == 3 / 5 and r["sure_wrong_all"] == 2 / 5


def test_an_errored_answer_is_dropped_and_its_retry_counts_once(tmp_path):
    decisions = [{"episode": "ep1", "question": "leak_capsule", "truth": "good"}]
    arms = [
        {"arm": "x", "index_global": 0, "probs": {"leaker": 0.5, "good": 0.5}, "error": "URLError"},
        {"arm": "x", "index_global": 0, "probs": {"leaker": 0.1, "good": 0.9}},
    ]
    rows = rep.rows_for(write_log(tmp_path, decisions, arms))["x"]
    assert len(rows) == 1
    assert rep.rates(rep.counts(rows))["false_alarms"] == 0.0


def test_the_interval_holds_the_point_value(tmp_path):
    decisions = [{"episode": f"ep{i // 4}", "question": "leak_capsule", "truth": "leaker" if i % 2 else "good"} for i in range(40)]
    arms = [{"arm": "x", "index_global": i, "probs": {"leaker": 0.7, "good": 0.3} if i % 3 else {"leaker": 0.2, "good": 0.8}} for i in range(40)]
    out = rep.report(rep.rows_for(write_log(tmp_path, decisions, arms))["x"], resamples=300)
    for k in ("caught", "false_alarms"):
        v, (lo, hi) = out[k]
        assert lo <= v <= hi
