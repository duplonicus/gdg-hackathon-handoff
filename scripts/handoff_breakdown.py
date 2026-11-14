"""Where a frozen ladder hands off and acts wrong, split by episode fault type and question (a re-cut, not a new claim).

Reads the frozen ladder (scripts/freeze_ladder.py freeze) and a log its arms were replayed on, applies the frozen
thresholds exactly as `confirm` does, and counts per fault type (clean, drop_on_carry, crowded): hand-offs per
decision and per episode, episodes with no judgement hand-off, and per question the hand-off and acted-wrong rates,
with acted-wrong split by direction (the machine said yes/good when the truth was no/defective, and the reverse).
Then, over every fault type, the in_bin decisions whose truth is "no", split by where the part was (`context.parts[part].in`,
the logged geometry check: in the other bin, or in no bin) and by what the ladder did (said yes / handed off / said no).

    .venv\\Scripts\\python.exe scripts\\handoff_breakdown.py --log output\\sort\\test04 --frozen output\\sort\\test03\\frozen-ladder.json

Writes <log>/handoff-breakdown.json.
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

from handoff.benchmark import collapse_finish, compose_ladder

RULE_DECIDED = "safe_to_grasp"


def where_part_was(d: dict) -> str:
    """For an in_bin decision: "wrong bin" (in a bin, not the intended one), "no bin", or "intended bin" (the logged check disagrees with the truth)."""
    ctx = d.get("context", {})
    bins_in = ctx.get("parts", {}).get(ctx.get("part"), {}).get("in") or []
    if not bins_in:
        return "no bin"
    return "intended bin" if ctx.get("intended_bin") in bins_in else "wrong bin"


def in_bin_misses(decisions: list[dict], rungs: list) -> dict:
    """in_bin decisions with truth "no", by where the part was, counted by what the ladder did."""
    miss = [d for d in decisions if d["question"] == "in_bin" and d.get("truth") == "no"]
    rows = compose_ladder(miss, rungs)  # one row per decision, in order: every one has a truth
    out: dict = {}
    for d, r in zip(miss, rows, strict=True):
        did = "handed off" if r["human"] else f"said {r['answer']}"
        cell = out.setdefault(where_part_was(d), {"said yes": 0, "handed off": 0, "said no": 0, "said yes at 0.9 or more": 0})
        cell[did] += 1
        cell["said yes at 0.9 or more"] += did == "said yes" and r["confidence"] >= 0.9
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--frozen", required=True)
    a = p.parse_args()
    log, frozen = Path(a.log), json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    fault = {}
    for line in (log / "episodes.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        fault[f"ep{e['seed']:04d}"] = e.get("fault") or "clean"
    rows = compose_ladder(decisions, [(r, arms[r], frozen["thresholds"][r]) for r in frozen["rungs"]])

    by_fault: dict = {}
    for f in sorted(set(fault.values())):
        mine = [r for r in rows if fault[r["episode"]] == f]
        judg = [r for r in mine if r["question"] != RULE_DECIDED]
        per_ep = Counter(r["episode"] for r in judg if r["human"])
        episodes = sorted({r["episode"] for r in mine})
        questions = {}
        for q in sorted({r["question"] for r in mine}):
            qr = [r for r in mine if r["question"] == q]
            wrong = [r for r in qr if not r["human"] and r["answer"] != r["truth"]]
            questions[q] = {
                "n": len(qr),
                "handed_to_person": round(sum(r["human"] for r in qr) / len(qr), 4),
                "acted_wrong": round(len(wrong) / len(qr), 4),
                "confidently_wrong": round(sum(r["confidence"] >= 0.9 for r in wrong) / len(qr), 4),
                "acted_wrong_by_direction": dict(Counter(f"said {r['answer']}, truth {r['truth']}" for r in wrong)),
            }
        by_fault[f] = {
            "episodes": len(episodes),
            "judgement_decisions": len(judg),
            "judgement_handed_to_person": round(sum(r["human"] for r in judg) / max(len(judg), 1), 4),
            "judgement_handoffs_per_episode": round(sum(per_ep.values()) / len(episodes), 3),
            "episodes_with_no_judgement_handoff": round(sum(per_ep[e] == 0 for e in episodes) / len(episodes), 3),
            "by_question": questions,
        }
    misses = in_bin_misses(decisions, [(r, arms[r], frozen["thresholds"][r]) for r in frozen["rungs"]])
    report = {"run": run_info(log=str(log), frozen=a.frozen), "ladder": frozen["rungs"], "thresholds": frozen["thresholds"], "by_fault": by_fault,
              "in_bin_truth_no_by_where": misses}
    for f, b in by_fault.items():
        print(f"{f}: {b['episodes']} episodes, judgement handed to person {b['judgement_handed_to_person']:.3f}, "
              f"{b['judgement_handoffs_per_episode']:.2f} per episode, {b['episodes_with_no_judgement_handoff']:.0%} of episodes with none")
        for q, v in b["by_question"].items():
            print(f"    {q:14s} n {v['n']:4d}  person {v['handed_to_person']:.3f}  acted wrong {v['acted_wrong']:.3f}  conf.wrong {v['confidently_wrong']:.3f}  {v['acted_wrong_by_direction']}")
    print("in_bin, truth no, by where the part was:")
    for where, c in misses.items():
        print(f"    {where:12s} {c}")
    (log / "handoff-breakdown.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {log / 'handoff-breakdown.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
