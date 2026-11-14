"""Choose the ladder to freeze for test04: every cheap-to-expensive chain of the strongest arms, scored like `robust`.

For each candidate ladder and threshold fit, thresholds are fitted on each split's held-out episodes and the ladder is
scored on that split's test episodes (test03 only; test04 is never read). The value of a decision is the same objective
the thresholds are fitted on: right alone +1, wrong alone -1, handed to the person 1 - human_cost. The pick is the
ladder with the highest mean value on the judgement questions (all but safe_to_grasp, which a hard rule decides); a
ladder within 0.005 of the best that has fewer rungs, then a lower median machine latency, wins the tie.

    .venv\\Scripts\\python.exe scripts\\pick_ladder.py --log output\\sort\\test03 --human-cost 0.2

Writes <log>/ladder-pick.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import load, load_arms, run_info

from handoff.benchmark import (
    choose_thresholds,
    choose_thresholds_per_question,
    collapse_finish,
    compose_ladder,
    split,
)

RULE_DECIDED = "safe_to_grasp"


def value(rows: list[dict], human_cost: float) -> float:
    if not rows:
        return 0.0
    return sum(1.0 - human_cost if r["human"] else (1.0 if r["answer"] == r["truth"] else -1.0) for r in rows) / len(
        rows
    )


def rates(rows: list[dict]) -> dict[str, float]:
    n = len(rows) or 1
    human = sum(r["human"] for r in rows)
    wrong = sum(1 for r in rows if not r["human"] and r["answer"] != r["truth"])
    cw = sum(1 for r in rows if not r["human"] and r["answer"] != r["truth"] and r["confidence"] >= 0.9)
    return {
        "human_rate": human / n,
        "acted_wrong": wrong / n,
        "confidently_wrong": cw / n,
        "alone_right": (n - human - wrong) / n,
    }


def pick_ladder(decisions: list[dict], arms: dict, *, hc: float, splits_n: int, held_out: float, seed0: int, min_decisions: int, pool_n: int, max_rungs: int, verbose: bool = True) -> dict:
    """Every ladder of the pool's arms x every fit, scored over the splits; returns pool, candidates, best and pick."""
    all_ids = {int(d["index_global"]) for d in decisions if d.get("truth") is not None}
    full = [
        n for n, ans in arms.items() if len(all_ids & set(ans)) >= 0.95 * len(all_ids)
    ]  # cloud subsets cannot be rungs
    latency = {
        n: statistics.median([r["latency_ms"] for r in arms[n].values() if r.get("latency_ms")] or [0.0]) for n in full
    }
    fits = {
        "per rung": choose_thresholds,
        "per question": choose_thresholds_per_question,
        f"per question, under {min_decisions} pooled": lambda d, r, **kw: choose_thresholds_per_question(
            d, r, min_decisions=min_decisions, **kw
        ),
    }
    splits = [split(decisions, held_out, seed0 + k) for k in range(splits_n)]

    def evaluate(rungs: tuple[str, ...]) -> dict:
        out = {}
        for fit_name, fit in fits.items():
            per = []
            for held, test in splits:
                t = fit(held, [(r, arms[r]) for r in rungs], human_cost=hc)
                rows = compose_ladder(test, [(r, arms[r], t[r]) for r in rungs])
                judg = [r for r in rows if r["question"] != RULE_DECIDED]
                per.append(
                    {
                        "value": value(rows, hc),
                        "judgement_value": value(judg, hc),
                        **rates(rows),
                        **{f"judgement_{k}": v for k, v in rates(judg).items()},
                    }
                )
            out[fit_name] = {k: round(statistics.mean(x[k] for x in per), 4) for k in per[0]} | {
                "judgement_value_min": round(min(x["judgement_value"] for x in per), 4),
                "judgement_value_max": round(max(x["judgement_value"] for x in per), 4),
            }
        return out

    solo = {n: evaluate((n,)) for n in full}
    best_solo = lambda n: max(f["judgement_value"] for f in solo[n].values())
    pool = sorted(full, key=best_solo, reverse=True)[: pool_n]
    if verbose:
        print(f"{len(decisions)} decisions, {splits_n} splits, human_cost {hc}; rung pool (best alone first):")
        for n in pool:
            print(f"  {n:28s} judgement value {best_solo(n):.4f}  median {latency[n]:.0f} ms")

    results = []
    for k in range(1, max_rungs + 1):
        for combo in itertools.combinations(pool, k):
            rungs = tuple(sorted(combo, key=lambda n: latency[n]))  # cheapest rung asked first
            ev = solo[rungs[0]] if k == 1 else evaluate(rungs)
            for fit_name, s in ev.items():
                results.append(
                    {"rungs": list(rungs), "fit": fit_name, "latency_ms": [round(latency[r], 1) for r in rungs], **s}
                )
    best = max(r["judgement_value"] for r in results)
    near = [r for r in results if r["judgement_value"] >= best - 0.005]
    pick = min(near, key=lambda r: (len(r["rungs"]), r["latency_ms"][0], -r["judgement_value"]))
    results.sort(key=lambda r: -r["judgement_value"])

    if verbose:
        print(f"\n{len(results)} ladder x fit candidates; top 15 by judgement value:")
        for r in results[:15]:
            print(
                f"  {r['judgement_value']:.4f} [{r['judgement_value_min']:.3f} {r['judgement_value_max']:.3f}]  acted wrong {r['judgement_acted_wrong']:.3f}  person {r['judgement_human_rate']:.3f}  {r['fit']:24s} {' > '.join(r['rungs'])}"
            )
    return {"pool": pool, "best_solo": best_solo, "latency": latency, "results": results, "best": best, "pick": pick}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--human-cost", type=float, default=0.2)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--held-out", type=float, default=0.3)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--min-decisions", type=int, default=50)
    p.add_argument("--pool", type=int, default=10, help="how many arms (best alone, by the same value) may be rungs")
    p.add_argument("--max-rungs", type=int, default=3)
    a = p.parse_args()
    log, hc = Path(a.log), a.human_cost

    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    found = pick_ladder(decisions, arms, hc=hc, splits_n=a.splits, held_out=a.held_out, seed0=a.seed0, min_decisions=a.min_decisions, pool_n=a.pool, max_rungs=a.max_rungs)
    pool, best_solo, latency, results, best, pick = (found[k] for k in ("pool", "best_solo", "latency", "results", "best", "pick"))
    print(
        f"\npick (within 0.005 of best {best:.4f}, fewest rungs, cheapest first rung): {' > '.join(pick['rungs'])} > human, {pick['fit']}"
    )
    report = {
        "run": run_info(
            human_cost=hc,
            splits=a.splits,
            held_out=a.held_out,
            seed0=a.seed0,
            min_decisions=a.min_decisions,
            pool=a.pool,
            max_rungs=a.max_rungs,
        ),
        "rule": "highest mean judgement value over splits; within 0.005 of it, fewest rungs, then cheapest first rung",
        "pool": [
            {"arm": n, "judgement_value": round(best_solo(n), 4), "median_ms": round(latency[n], 1)} for n in pool
        ],
        "pick": pick,
        "candidates": results,
    }
    (log / "ladder-pick.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {log / 'ladder-pick.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
