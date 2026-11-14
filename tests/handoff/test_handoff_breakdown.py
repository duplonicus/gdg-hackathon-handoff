"""handoff_breakdown.py: fault types join on zero-padded episode names, and in_bin misses split by where the part was."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("handoff_breakdown", SCRIPTS / "handoff_breakdown.py")
hb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hb)


def in_bin(i, episode, where, intended="good"):
    return {"index_global": i, "episode": episode, "question": "in_bin", "truth": "no",
            "context": {"part": "p", "intended_bin": intended, "parts": {"p": {"in": where}}}}


def yes(conf):
    return {"probs": {"yes": conf, "no": 1 - conf}, "latency_ms": 1.0}


def test_in_bin_misses_split_wrong_bin_from_no_bin():
    decs = [in_bin(0, "ep0400", ["rework"]), in_bin(1, "ep0400", []), in_bin(2, "ep0401", []), in_bin(3, "ep0401", ["good"])]
    answers = {0: yes(0.99), 1: yes(0.5), 2: yes(0.1), 3: yes(0.85)}
    out = hb.in_bin_misses(decs, [("d3", answers, 0.8)])
    assert out["wrong bin"] == {"said yes": 1, "handed off": 0, "said no": 0, "said yes at 0.9 or more": 1}
    assert out["no bin"] == {"said yes": 0, "handed off": 1, "said no": 1, "said yes at 0.9 or more": 0}
    assert out["intended bin"]["said yes"] == 1  # the logged check disagrees with the truth: kept visible, not folded in


def test_episodes_under_seed_1000_join_their_fault(tmp_path, monkeypatch):
    decs = [in_bin(0, "ep0400", ["rework"]), {"index_global": 1, "episode": "ep1000", "question": "finish", "truth": "good"}]
    arms = {"d3": {0: yes(0.95), 1: {"probs": {"good": 0.9, "defective": 0.1}, "latency_ms": 1.0}}}
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"seed": 400, "fault": None}) + "\n" + json.dumps({"seed": 1000, "fault": "crowded"}) + "\n", encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"rungs": ["d3"], "thresholds": {"d3": 0.8}}), encoding="utf-8")
    monkeypatch.setattr(hb, "load", lambda log: decs)
    monkeypatch.setattr(hb, "load_arms", lambda log: arms)
    monkeypatch.setattr(hb, "collapse_finish", lambda d, a: None)
    monkeypatch.setattr(sys, "argv", ["handoff_breakdown.py", "--log", str(tmp_path), "--frozen", str(frozen)])
    assert hb.main() == 0
    report = json.loads((tmp_path / "handoff-breakdown.json").read_text(encoding="utf-8"))
    assert report["by_fault"]["clean"]["episodes"] == 1 and report["by_fault"]["crowded"]["episodes"] == 1
    assert report["in_bin_truth_no_by_where"]["wrong bin"]["said yes"] == 1
