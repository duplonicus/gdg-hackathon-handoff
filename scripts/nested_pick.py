"""Measure how optimistic pick_ladder.py's "prediction" is (winner's curse), with a nested split of one log.

The prediction frozen for test04 is the picked ladder's own score over the splits it was picked on, so it is biased
upward. Here each outer split cuts the log by episode into a selection part and an evaluation part. The whole pick
(every ladder x fit, over its inner splits) runs on the selection part only; the picked ladder is then fitted on all
of the selection part, as `freeze_ladder.py freeze` does, and scored once on the evaluation part, which the pick never
saw. Predicted minus observed, averaged over the outer splits, is the bias.

    .venv\\Scripts\\python.exe scripts\\nested_pick.py --log output\\sort\\test03

Writes <log>/nested-pick.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import load, load_arms, run_info
from freeze_ladder import fit
from pick_ladder import RULE_DECIDED, pick_ladder, rates, value

from handoff.benchmark import collapse_finish, compose_ladder, split

RATES = ("judgement_value", "judgement_acted_wrong", "judgement_confidently_wrong", "judgement_human_rate", "judgement_alone_right", "acted_wrong", "human_rate")


def observed(rows: list[dict], hc: float) -> dict[str, float]:
    judg = [r for r in rows if r["question"] != RULE_DECIDED]
    return {"judgement_value": value(judg, hc), **rates(rows), **{f"judgement_{k}": v for k, v in rates(judg).items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--outer", type=int, default=10, help="outer splits")
    p.add_argument("--evaluate", type=float, default=0.3, help="fraction of episodes the pick never sees")
    p.add_argument("--outer-seed0", type=int, default=1000)
    p.add_argument("--human-cost", type=float, default=0.2)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--held-out", type=float, default=0.3)
    p.add_argument("--min-decisions", type=int, default=50)
    p.add_argument("--pool", type=int, default=10)
    p.add_argument("--max-rungs", type=int, default=3)
    a = p.parse_args()
    log, hc = Path(a.log), a.human_cost
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)

    outer = []
    for k in range(a.outer):
        held_back, selection = split(decisions, a.evaluate, a.outer_seed0 + k)  # held_back: never seen by the pick
        found = pick_ladder(selection, arms, hc=hc, splits_n=a.splits, held_out=a.held_out, seed0=0, min_decisions=a.min_decisions, pool_n=a.pool, max_rungs=a.max_rungs, verbose=False)
        pick = found["pick"]
        thresholds = fit(pick["fit"], a.min_decisions)(selection, [(r, arms[r]) for r in pick["rungs"]], human_cost=hc)
        obs = observed(compose_ladder(held_back, [(r, arms[r], thresholds[r]) for r in pick["rungs"]]), hc)
        row = {"outer": k, "ladder": pick["rungs"], "fit": pick["fit"], "predicted": {m: pick[m] for m in RATES}, "observed": {m: round(obs[m], 4) for m in RATES}}
        outer.append(row)
        print(f"outer {k}: {' > '.join(pick['rungs'])} ({pick['fit']}): judgement value {pick['judgement_value']:.3f} predicted, {obs['judgement_value']:.3f} observed; "
              f"judgement acted wrong {pick['judgement_acted_wrong']:.3f} predicted, {obs['judgement_acted_wrong']:.3f} observed")
    bias = {}
    for m in RATES:
        diffs = [r["predicted"][m] - r["observed"][m] for r in outer]
        bias[m] = {"mean_predicted_minus_observed": round(statistics.mean(diffs), 4), "sd": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None}
    print("\npredicted minus observed, mean (sd) over outer splits:")
    for m, b in bias.items():
        print(f"  {m:30s} {b['mean_predicted_minus_observed']:+.4f} ({b['sd']})")
    report = {"run": run_info(**{k: v for k, v in vars(a).items() if k != "log"}, log=str(log)), "bias": bias, "outer": outer}
    (log / "nested-pick.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {log / 'nested-pick.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
