"""Rescore the test04 confirmation the way the running system decides: hard rules answer before any model.

The confirmation (freeze_ladder.py confirm) composed the frozen ladder over every logged decision, including the ones
the sim's hard rules answered while the log was made (``answered_by == "rule"``: a part on the floor or a crowded
station forces safe_to_grasp to abort; a part on the floor forces sort_complete to no). In the running system the
ladder never sees those: Ladder.decide returns the rule's answer before d3 is asked. This scores both ways, side by
side, with the same frozen thresholds and the same 95% by-episode bootstrap, and counts what moved. It changes no
frozen value; it is a post-confirmation note, like the winner's-curse check.

    .venv\\Scripts\\python.exe scripts\\live_equivalent.py --log output\\sort\\test04 --frozen output\\sort\\test03\\frozen-ladder.json

Writes <log>/live-equivalent.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import load, load_arms, run_info

from handoff.benchmark import collapse_finish, compose_ladder, score


def rule_decided(rows: list[dict], decisions: list[dict]) -> list[dict]:
    """Replace each row whose decision a hard rule answered with the rule's own answer (not a hand-off, not d3's)."""
    labelled = [d for d in decisions if d.get("truth") is not None]
    assert len(labelled) == len(rows), "compose_ladder keeps one row per labelled decision, in order"
    out = []
    for d, r in zip(labelled, rows, strict=True):
        if d.get("answered_by") == "rule":
            out.append({**r, "answer": d["answer"], "confidence": 1.0, "human": False, "rung": "rule", "latency_ms": 0.0})
        else:
            out.append(r)
    return out


def rates(rep) -> dict:
    iv = rep.interval or {}

    def block(prefix: str, r: dict) -> dict:
        p = f"{prefix}_" if prefix else ""
        lo, hi = iv.get(f"{p}accuracy", (float("nan"), float("nan")))
        return {
            "n": r.get("n", rep.n),
            "acted_wrong": [round(1 - r["accuracy"], 4), [round(1 - hi, 4), round(1 - lo, 4)]],
            "confidently_wrong": [round(r["confidently_wrong"], 4), [round(x, 4) for x in iv.get(f"{p}confidently_wrong", ())]],
            "handed_to_person": [round(r["human_rate"], 4), [round(x, 4) for x in iv.get(f"{p}human_rate", ())]],
            "alone_right": [round(r["alone_right"], 4), [round(x, 4) for x in iv.get(f"{p}alone_right", ())]],
        }

    overall = {"accuracy": rep.accuracy, "confidently_wrong": rep.confidently_wrong, "human_rate": rep.human_rate, "alone_right": rep.alone_right}
    return {"overall": block("", overall), "judgement": block("judgement", rep.judgement or {})}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--frozen", required=True)
    a = p.parse_args()
    log, frozen = Path(a.log), json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    rungs = frozen["rungs"]
    replayed = compose_ladder(decisions, [(r, arms[r], frozen["thresholds"][r]) for r in rungs])
    live = rule_decided(replayed, decisions)

    labelled = [d for d in decisions if d.get("truth") is not None]
    moved = Counter()
    for d, r in zip(labelled, replayed, strict=True):
        if d.get("answered_by") == "rule":
            was = "handed to person" if r["human"] else ("acted wrong" if r["answer"] != r["truth"] else "alone right")
            moved[(d["question"], was)] += 1
    rule_wrong = sum(1 for d in labelled if d.get("answered_by") == "rule" and d["answer"] != d["truth"])

    report = {
        "run": run_info(frozen=a.frozen, n_decisions=len(decisions)),
        "ladder": rungs,
        "thresholds": frozen["thresholds"],
        "format": "each rate: [value, 95% by-episode interval]",
        "rule_decided": {f"{q}: replay had {was}": n for (q, was), n in sorted(moved.items())},
        "rule_answers_wrong": rule_wrong,
        "as_replayed": rates(score(" > ".join(rungs), replayed)),
        "live_equivalent": rates(score(" > ".join(rungs), live)),
    }
    print("rule-decided decisions, by what the replay had scored them as:")
    for k, n in report["rule_decided"].items():
        print(f"  {k}: {n}")
    print(f"rule answers that disagree with the truth: {rule_wrong}")
    for scope in ("overall", "judgement"):
        print(f"\n{scope}, n {report['as_replayed'][scope]['n']}:   as replayed [95%]      live-equivalent [95%]")
        for k in ("acted_wrong", "confidently_wrong", "handed_to_person", "alone_right"):
            (v1, c1), (v2, c2) = report["as_replayed"][scope][k], report["live_equivalent"][scope][k]
            print(f"  {k:18s} {v1:.3f} [{c1[0]:.3f} {c1[1]:.3f}]   {v2:.3f} [{c2[0]:.3f} {c2[1]:.3f}]")
    (log / "live-equivalent.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten {log / 'live-equivalent.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
