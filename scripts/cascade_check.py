"""Exploratory (not a claim): dup's 2 > 4 > 9 > 27 idea, scored on test03 the same way pick_ladder.py scored its pool.

pick_ladder.py only let the 10 best arms alone be rungs, so no 2B or 4B model was ever tried as a first rung. This
scores fixed small-first ladders with pick_ladder's own value, rates, splits (10, held-out 0.3, seeds 0 to 9) and
threshold fits (the best of the three kept per ladder), human_cost 0.2, judgement questions. Results: VALIDATION.md
"Exploratory: small-model-first cascades".

    .venv\\Scripts\\python.exe scripts\\cascade_check.py      (CPU only, reads output/sort/test03)
"""
import statistics, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from benchmark import load, load_arms
from handoff.benchmark import choose_thresholds, choose_thresholds_per_question, collapse_finish, compose_ladder, split
from pick_ladder import RULE_DECIDED, rates, value

log = ROOT / "output" / "sort" / "test03"; hc = 0.2
decisions = load(log); arms = load_arms(log); collapse_finish(decisions, arms)
lat = {n: statistics.median([r["latency_ms"] for r in a.values() if r.get("latency_ms")] or [0]) for n, a in arms.items()}
splits = [split(decisions, 0.3, k) for k in range(10)]
fits = {"per rung": choose_thresholds, "per question": choose_thresholds_per_question,
        "per question, under 50 pooled": lambda d, r, **kw: choose_thresholds_per_question(d, r, min_decisions=50, **kw)}
LADDERS = [
    ("d3-27b-vision",),
    ("d3-nano-vision", "d3-mini-vision", "d3-flash-vision", "d3-27b-vision"),
    ("d3-nano-vision", "d3-27b-vision"), ("d3-mini-vision", "d3-27b-vision"), ("d3-flash-vision", "d3-27b-vision"),
    ("d3-nano-vision", "d3-mini-vision", "d3-27b-vision"), ("d3-nano-vision", "d3-flash-vision", "d3-27b-vision"),
    ("intern-2b-vision", "intern-4b-vision", "d3-27b-vision"), ("intern-2b-vision", "d3-27b-vision"), ("intern-4b-vision", "d3-27b-vision"),
    ("decider-2b-vision", "d3-27b-vision"),
]
for rungs in LADDERS:
    best = None
    for fname, fit in fits.items():
        per, share = [], Counter()
        for held, test in splits:
            t = fit(held, [(r, arms[r]) for r in rungs], human_cost=hc)
            rows = compose_ladder(test, [(r, arms[r], t[r]) for r in rungs])
            judg = [r for r in rows if r["question"] != RULE_DECIDED]
            share.update(r["rung"] for r in judg)
            per.append({"jv": value(judg, hc), **{f"j_{k}": v for k, v in rates(judg).items()}, "ms": statistics.mean(r["latency_ms"] for r in judg if not r["human"]) if any(not r["human"] for r in judg) else 0})
        m = {k: statistics.mean(x[k] for x in per) for k in per[0]}
        if best is None or m["jv"] > best[1]["jv"]:
            tot = sum(share.values())
            best = (fname, m, {k: round(v / tot, 3) for k, v in share.items()})
    f, m, sh = best
    print(f"{m['jv']:.4f}  person {m['j_human_rate']:.3f}  wrong {m['j_acted_wrong']:.4f}  machine ms {m['ms']:.0f}  [{f}]  {' > '.join(r.replace('-vision','') for r in rungs)}")
    print(f"        answered by: {sh}")
