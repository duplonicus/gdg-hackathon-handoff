"""live_equivalent.py: rows a hard rule decided are scored as the rule's answer, not as d3's or as a hand-off."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("live_equivalent", SCRIPTS / "live_equivalent.py")
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def test_rule_decided_rows_take_the_rules_answer_and_others_stay():
    decisions = [
        {"question": "safe_to_grasp", "truth": "abort", "answer": "abort", "answered_by": "rule"},
        {"question": "sort_complete", "truth": "no", "answer": "no", "answered_by": "rule"},
        {"question": "in_bin", "truth": "yes", "answer": "yes", "answered_by": "oracle"},
        {"question": "finish", "truth": None, "answer": "good", "answered_by": "rule"},  # unlabelled: no row at all
    ]
    replayed = [
        {"question": "safe_to_grasp", "answer": "grasp", "truth": "abort", "confidence": 0.95, "human": False, "rung": "d3"},  # d3 acted wrong
        {"question": "sort_complete", "answer": "no", "truth": "no", "confidence": 1.0, "human": True, "rung": "human"},  # a hand-off
        {"question": "in_bin", "answer": "no", "truth": "yes", "confidence": 0.99, "human": False, "rung": "d3"},
    ]
    out = live.rule_decided(replayed, decisions)
    assert [(r["rung"], r["answer"], r["human"]) for r in out] == [("rule", "abort", False), ("rule", "no", False), ("d3", "no", False)]
    assert out[2] is replayed[2]
