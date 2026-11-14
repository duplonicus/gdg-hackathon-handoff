"""nested_pick.py: the pick and its threshold fit never see the episodes the picked ladder is scored on."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("nested_pick", SCRIPTS / "nested_pick.py")
nested = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nested)


def test_the_pick_never_sees_the_evaluation_episodes(tmp_path, monkeypatch):
    decs = [{"index_global": i, "episode": f"ep{i // 3:04d}", "question": "finish", "truth": "good"} for i in range(60)]
    arms = {"a": {i: {"probs": {"good": 0.95, "defective": 0.05}, "latency_ms": 1.0} for i in range(60)}}
    seen_by_pick, seen_by_fit, scored = [], [], []
    pick = {"rungs": ["a"], "fit": "per rung", **{m: 0.5 for m in nested.RATES}}

    def fake_pick(selection, arms, **kw):
        seen_by_pick.append({d["episode"] for d in selection})
        return {"pick": pick}

    def fake_fit(name, min_decisions):
        return lambda selection, rungs, **kw: seen_by_fit.append({d["episode"] for d in selection}) or {"a": 0.9}

    real_compose = nested.compose_ladder
    monkeypatch.setattr(nested, "pick_ladder", fake_pick)
    monkeypatch.setattr(nested, "fit", fake_fit)
    monkeypatch.setattr(nested, "compose_ladder", lambda d, r: scored.append({x["episode"] for x in d}) or real_compose(d, r))
    monkeypatch.setattr(nested, "load", lambda log: decs)
    monkeypatch.setattr(nested, "load_arms", lambda log: arms)
    monkeypatch.setattr(nested, "collapse_finish", lambda d, a: None)
    monkeypatch.setattr(sys, "argv", ["nested_pick.py", "--log", str(tmp_path), "--outer", "3"])
    assert nested.main() == 0
    for picked, fitted, evaluated in zip(seen_by_pick, seen_by_fit, scored, strict=True):
        assert picked == fitted and not picked & evaluated and len(picked | evaluated) == 20
    report = json.loads((tmp_path / "nested-pick.json").read_text(encoding="utf-8"))
    assert report["bias"]["judgement_acted_wrong"]["mean_predicted_minus_observed"] == 0.5  # predicted 0.5, every answer right
